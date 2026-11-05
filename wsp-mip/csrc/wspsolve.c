/* wspsolve — the weight-space-programmed branching rule as a native SCIP plugin.
 *
 * Same features, net and lookahead policy as branching.py, without the Python overhead, so the
 * comparison against SCIP's own C rules is fair on time. Built-in rules run from the same binary
 * (same SCIP build) for pairing.
 *
 * usage: wspsolve INSTANCE RULE TIMELIMIT [NET.txt]
 *   RULE = net | relpscost | fullstrong | pscost | mostinf ...
 *   env SCIP_DEFAULTS=1 keeps SCIP's factory settings; otherwise Gasse et al. (no restarts, cuts
 *   only at the root).
 * prints one JSON line: instance, rule, status, nodes, time, gap, obj.
 *
 * NET.txt (written by nets2txt.py): "inputs nlayers" / "k maxdepth itlim" / per layer
 * "rows cols act" (act 0 linear, 1 relu, 2 tanh), then rows*cols weights, then rows biases.
 */
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "scip/scip.h"
#include "scip/scipdefplugins.h"

#define NFEAT 11
#define MAXLAYERS 8

typedef struct {
   int rows, cols, act;
   double* w;
   double* b;
} Layer;

typedef struct {
   int inputs, nlayers;
   int la_k, la_maxdepth, la_itlim;
   Layer layer[MAXLAYERS];
} Net;

static Net* load_net(const char* path) {
   FILE* f = fopen(path, "r");
   if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
   Net* net = calloc(1, sizeof(Net));
   if (fscanf(f, "%d %d", &net->inputs, &net->nlayers) != 2) exit(2);
   if (fscanf(f, "%d %d %d", &net->la_k, &net->la_maxdepth, &net->la_itlim) != 3) exit(2);
   for (int l = 0; l < net->nlayers; ++l) {
      Layer* L = &net->layer[l];
      if (fscanf(f, "%d %d %d", &L->rows, &L->cols, &L->act) != 3) exit(2);
      L->w = malloc(sizeof(double) * L->rows * L->cols);
      L->b = malloc(sizeof(double) * L->rows);
      for (int i = 0; i < L->rows * L->cols; ++i) if (fscanf(f, "%lf", &L->w[i]) != 1) exit(2);
      for (int i = 0; i < L->rows; ++i) if (fscanf(f, "%lf", &L->b[i]) != 1) exit(2);
   }
   fclose(f);
   return net;
}

static double net_eval(const Net* net, const double* x) {
   double buf[2][256];
   int width = net->inputs;
   memcpy(buf[0], x, sizeof(double) * width);
   int cur = 0;
   for (int l = 0; l < net->nlayers; ++l) {
      const Layer* L = &net->layer[l];
      for (int r = 0; r < L->rows; ++r) {
         double s = L->b[r];
         for (int c = 0; c < L->cols; ++c) s += L->w[r * L->cols + c] * buf[cur][c];
         if (L->act == 1) s = s > 0 ? s : 0;
         else if (L->act == 2) s = tanh(s);
         buf[1 - cur][r] = s;
      }
      cur = 1 - cur;
      width = L->rows;
   }
   return buf[cur][0];
}

struct SCIP_BranchruleData {
   Net* net;
   /* scratch, grown on demand */
   int rowcap, candcap;
   int* rowcount;     /* fractional candidates per LP row */
   int* rowstart;     /* CSR: row -> candidates */
   int* rowcands;
   int* stamp;
   double* feat;
   double* score;
   int redvar[64];
   int reddir[64];
};

static void ensure(SCIP_BRANCHRULEDATA* d, int nrows, int ncands, int nnz) {
   if (nrows > d->rowcap) {
      d->rowcap = 2 * nrows;
      d->rowcount = realloc(d->rowcount, sizeof(int) * d->rowcap);
      d->rowstart = realloc(d->rowstart, sizeof(int) * (d->rowcap + 1));
   }
   if (ncands > d->candcap) {
      d->candcap = 2 * ncands;
      d->stamp = realloc(d->stamp, sizeof(int) * d->candcap);
      d->feat = realloc(d->feat, sizeof(double) * d->candcap * NFEAT);
      d->score = realloc(d->score, sizeof(double) * d->candcap);
   }
   d->rowcands = realloc(d->rowcands, sizeof(int) * (nnz + 1));
}

static SCIP_DECL_BRANCHEXECLP(branchExeclpWsp) {
   SCIP_BRANCHRULEDATA* d = SCIPbranchruleGetData(branchrule);
   SCIP_VAR** cands;
   SCIP_Real* sols;
   int ncands, nprio;
   SCIP_CALL(SCIPgetLPBranchCands(scip, &cands, &sols, NULL, &ncands, &nprio, NULL));
   int n = nprio;
   if (n == 0) { *result = SCIP_DIDNOTRUN; return SCIP_OKAY; }

   int nlprows = SCIPgetNLPRows(scip);
   int nnz = 0;
   for (int k = 0; k < n; ++k) nnz += SCIPcolGetNLPNonz(SCIPvarGetCol(cands[k]));
   ensure(d, nlprows, n, nnz);
   double* X = d->feat;

   /* per-candidate base features */
   memset(d->rowcount, 0, sizeof(int) * nlprows);
   for (int k = 0; k < n; ++k) {
      double v = sols[k], f = v - floor(v);
      SCIP_COL* col = SCIPvarGetCol(cands[k]);
      int nz = SCIPcolGetNLPNonz(col);
      SCIP_ROW** rows = SCIPcolGetRows(col);
      SCIP_Real* vals = SCIPcolGetVals(col);
      double ntight = 0, dualm = 0;
      for (int i = 0; i < nz; ++i) {
         SCIP_ROW* row = rows[i];
         double act = SCIPgetRowLPActivity(scip, row);
         if (fabs(act - SCIProwGetRhs(row)) < 1e-6 || fabs(act - SCIProwGetLhs(row)) < 1e-6) ntight += 1;
         dualm += fabs(SCIProwGetDualsol(row) * vals[i]);
         d->rowcount[SCIProwGetLPPos(row)]++;
      }
      double* x = X + k * NFEAT;
      x[0] = 2 * fmin(f, 1 - f);
      x[1] = SCIPgetVarPseudocostScore(scip, cands[k], v);
      x[2] = fabs(SCIPvarGetObj(cands[k]));
      x[3] = nz;
      x[4] = nz ? ntight / nz : 0.0;
      x[5] = dualm;
      x[6] = v;
   }
   /* CSR row -> candidates */
   d->rowstart[0] = 0;
   for (int r = 0; r < nlprows; ++r) d->rowstart[r + 1] = d->rowstart[r] + d->rowcount[r];
   memset(d->rowcount, 0, sizeof(int) * nlprows);
   for (int k = 0; k < n; ++k) {
      SCIP_COL* col = SCIPvarGetCol(cands[k]);
      int nz = SCIPcolGetNLPNonz(col);
      SCIP_ROW** rows = SCIPcolGetRows(col);
      for (int i = 0; i < nz; ++i) {
         int r = SCIProwGetLPPos(rows[i]);
         d->rowcands[d->rowstart[r] + d->rowcount[r]++] = k;
      }
   }
   /* structural features */
   for (int k = 0; k < n; ++k) d->stamp[k] = -1;
   double emin = DBL_MAX;
   int anypos = 0;
   for (int k = 0; k < n; ++k) {
      SCIP_COL* col = SCIPvarGetCol(cands[k]);
      int nz = SCIPcolGetNLPNonz(col);
      SCIP_ROW** rows = SCIPcolGetRows(col);
      double fconf = 0;
      int minm = INT_MAX, rowf = 0;
      for (int i = 0; i < nz; ++i) {
         int r = SCIProwGetLPPos(rows[i]);
         int m = d->rowcount[r];
         if (m < minm) minm = m;
         if (m > 1) rowf++;
         for (int p = d->rowstart[r]; p < d->rowstart[r + 1]; ++p) {
            int o = d->rowcands[p];
            if (o != k && d->stamp[o] != k) { d->stamp[o] = k; fconf += sols[o]; }
         }
      }
      if (nz == 0) minm = 1;
      double* x = X + k * NFEAT;
      x[7] = fconf;
      x[8] = minm > 1 ? 1.0 / (minm - 1) : 0.0;
      x[9] = rowf;
      x[10] = fabs(SCIPvarGetObj(cands[k])) / (nz > 0 ? nz : 1);
      if (x[10] < emin) emin = x[10];
      if (x[10] > 0) anypos = 1;
   }
   for (int k = 0; k < n; ++k) {
      double* x = X + k * NFEAT;
      if (!anypos) x[10] = 1.0;
      else x[10] = x[10] > 0 ? emin / fmax(x[10], 1e-12) : 1.0;
   }
   static const int norm[] = {1, 2, 3, 5, 7, 9};
   for (int c = 0; c < 6; ++c) {
      double mx = 0;
      for (int k = 0; k < n; ++k) if (X[k * NFEAT + norm[c]] > mx) mx = X[k * NFEAT + norm[c]];
      if (mx > 0) for (int k = 0; k < n; ++k) X[k * NFEAT + norm[c]] /= mx;
   }

   int best = 0;
   for (int k = 0; k < n; ++k) {
      d->score[k] = net_eval(d->net, X + k * NFEAT);
      if (d->score[k] > d->score[best]) best = k;
   }

   /* optional: strong-branch only the net's top-k candidates */
   const Net* net = d->net;
   if (net->la_k > 0 && n > 1 && SCIPgetDepth(scip) <= net->la_maxdepth) {
      int k = net->la_k < n ? net->la_k : n;
      int top[64];
      if (k > 64) k = 64;
      for (int t = 0; t < k; ++t) {           /* partial selection sort on score */
         int arg = -1;
         for (int j = 0; j < n; ++j) {
            int used = 0;
            for (int u = 0; u < t; ++u) if (top[u] == j) { used = 1; break; }
            if (!used && (arg < 0 || d->score[j] > d->score[arg])) arg = j;
         }
         top[t] = arg;
      }
      double lpobj = SCIPgetLPObjval(scip), bestsb = -1;
      int nred = 0;
      SCIP_Bool cutoff = FALSE;
      SCIP_CALL(SCIPstartStrongbranch(scip, FALSE));
      for (int t = 0; t < k; ++t) {
         SCIP_Real down, up;
         SCIP_Bool dvalid, uvalid, dinf, uinf, dconf, uconf, lperror;
         SCIP_CALL(SCIPgetVarStrongbranchFrac(scip, cands[top[t]], net->la_itlim, FALSE, &down, &up,
                                              &dvalid, &uvalid, &dinf, &uinf, &dconf, &uconf, &lperror));
         if (lperror) break;
         /* a child the LP proves infeasible (or past the incumbent) is a free domain reduction,
            as in SCIP's own strong branching; collect it and keep evaluating */
         if (dinf && uinf) { cutoff = TRUE; break; }
         if (dinf || uinf) { d->redvar[nred] = top[t]; d->reddir[nred] = dinf; nred++; continue; }
         double dg = dinf ? 1e9 : fmax(down - lpobj, 0.0);
         double ug = uinf ? 1e9 : fmax(up - lpobj, 0.0);
         double s = fmax(dg, 1e-6) * fmax(ug, 1e-6);
         if (s > bestsb) { bestsb = s; best = top[t]; }
      }
      SCIP_CALL(SCIPendStrongbranch(scip));
      if (cutoff) { *result = SCIP_CUTOFF; return SCIP_OKAY; }
      if (nred > 0) {
         for (int r = 0; r < nred; ++r) {
            SCIP_VAR* var = cands[d->redvar[r]];
            double v = sols[d->redvar[r]];
            if (d->reddir[r]) SCIP_CALL(SCIPchgVarLb(scip, var, ceil(v)));   /* down infeasible */
            else SCIP_CALL(SCIPchgVarUb(scip, var, floor(v)));               /* up infeasible */
         }
         *result = SCIP_REDUCEDDOM;
         return SCIP_OKAY;
      }
   }

   SCIP_CALL(SCIPbranchVar(scip, cands[best], NULL, NULL, NULL));
   *result = SCIP_BRANCHED;
   return SCIP_OKAY;
}

static SCIP_DECL_BRANCHFREE(branchFreeWsp) {
   SCIP_BRANCHRULEDATA* d = SCIPbranchruleGetData(branchrule);
   free(d->rowcount); free(d->rowstart); free(d->rowcands); free(d->stamp);
   free(d->feat); free(d->score);
   free(d);
   SCIPbranchruleSetData(branchrule, NULL);
   return SCIP_OKAY;
}

static double wall(void) {
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char** argv) {
   if (argc < 4) {
      fprintf(stderr, "usage: wspsolve INSTANCE RULE TIMELIMIT [NET.txt]\n");
      return 1;
   }
   const char* inst = argv[1];
   const char* rule = argv[2];
   double tlimit = atof(argv[3]);
   SCIP* scip;
   SCIP_CALL(SCIPcreate(&scip));
   SCIP_CALL(SCIPincludeDefaultPlugins(scip));
   SCIP_CALL(SCIPsetIntParam(scip, "display/verblevel", 0));
   if (!getenv("SCIP_DEFAULTS")) {
      SCIP_CALL(SCIPsetIntParam(scip, "presolving/maxrestarts", 0));
      SCIP_CALL(SCIPsetIntParam(scip, "separating/maxrounds", 0));
   }
   SCIP_CALL(SCIPsetRealParam(scip, "limits/time", tlimit));
   SCIP_CALL(SCIPsetIntParam(scip, "randomization/randomseedshift", 0));

   char rulename[256];
   if (strcmp(rule, "net") == 0) {
      if (argc < 5) { fprintf(stderr, "net rule needs NET.txt\n"); return 1; }
      SCIP_BRANCHRULEDATA* d = calloc(1, sizeof(SCIP_BRANCHRULEDATA));
      d->net = load_net(argv[4]);
      SCIP_BRANCHRULE* br;
      SCIP_CALL(SCIPincludeBranchruleBasic(scip, &br, "wsp", "weight-space programmed", 9999999,
                                           -1, 1.0, d));
      SCIP_CALL(SCIPsetBranchruleExecLp(scip, br, branchExeclpWsp));
      SCIP_CALL(SCIPsetBranchruleFree(scip, br, branchFreeWsp));
      const char* base = strrchr(argv[4], '/');
      base = base ? base + 1 : argv[4];
      snprintf(rulename, sizeof rulename, "c:%.*s", (int)(strcspn(base, ".")), base);
   } else {
      char p[256];
      snprintf(p, sizeof p, "branching/%s/priority", rule);
      SCIP_CALL(SCIPsetIntParam(scip, p, 9999999));
      snprintf(rulename, sizeof rulename, "c:%s", rule);
   }

   SCIP_CALL(SCIPreadProb(scip, inst, NULL));
   double t0 = wall();
   SCIP_CALL(SCIPsolve(scip));
   double t = wall() - t0;

   const char* st = "other";
   switch (SCIPgetStatus(scip)) {
      case SCIP_STATUS_OPTIMAL: st = "optimal"; break;
      case SCIP_STATUS_TIMELIMIT: st = "timelimit"; break;
      case SCIP_STATUS_INFEASIBLE: st = "infeasible"; break;
      default: break;
   }
   printf("{\"instance\": \"%s\", \"rule\": \"%s\", \"status\": \"%s\", \"nodes\": %lld, "
          "\"time\": %.3f, \"gap\": %.6g, ",
          inst, rulename, st, (long long)SCIPgetNNodes(scip), t, SCIPgetGap(scip));
   if (SCIPgetNSols(scip) > 0) printf("\"obj\": %.10g}\n", SCIPgetPrimalbound(scip));
   else printf("\"obj\": null}\n");
   SCIP_CALL(SCIPfree(&scip));
   return 0;
}
