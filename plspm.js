// plspm.js - Partial Least Squares Path Modeling in JavaScript

class PLSPM {
    constructor(maxIter = 100, tol = 1e-6, eval = true, plsc = false) {
        this.maxIter = maxIter;
        this.tol = tol;
        this.eval = eval;
        this.plsc = plsc;
    }

    // دوال مساعدة أساسية
    static correlate(Xk, yk) {
        const X = Array.isArray(Xk[0]) ? Xk.map(row => [...row]) : [Xk];
        const y = Array.isArray(yk) ? yk : [yk];

        const XMean = this.mean(X, 0);
        const yMean = this.mean(y);

        const Xm = X.map(row => row.map((val, i) => val - XMean[i]));
        const ym = y.map(val => val - yMean);

        const X2 = Xm[0].map((_, i) =>
            Xm.reduce((sum, row) => sum + row[i] * row[i], 0)
        );
        const y2 = ym.reduce((sum, val) => sum + val * val, 0);

        const XmNorm = Xm.map(row =>
            row.map((val, i) => X2[i] === 0 ? 0 : val / Math.sqrt(X2[i]))
        );
        const ymNorm = ym.map(val => y2 === 0 ? 0 : val / Math.sqrt(y2));

        const r = XmNorm[0].map((_, i) =>
            XmNorm.reduce((sum, row, j) => sum + row[i] * ymNorm[j], 0)
        );

        return r.length === 1 ? r[0] : r;
    }

    static solveLinearSystem(A, b) {
        // تنفيذ مبسط لحل نظام المعادلات الخطية
        const n = A.length;
        const m = A[0].length;

        if (n !== b.length) {
            throw new Error('أبعاد المصفوفة A والمتجه b غير متطابقة');
        }

        // طريقة المربعات الصغرى البسيطة
        const At = this.transpose(A);
        const AtA = this.multiplyMatrices(At, A);
        const Atb = this.multiplyMatrixVector(At, b);

        return this.solveSymmetricSystem(AtA, Atb);
    }

    static solveSymmetricSystem(A, b) {
        const n = A.length;
        const x = new Array(n).fill(0);

        // طريقة جاوس-سيدل المبسطة
        for (let iter = 0; iter < 100; iter++) {
            let maxDiff = 0;
            for (let i = 0; i < n; i++) {
                let sum = 0;
                for (let j = 0; j < n; j++) {
                    if (i !== j) {
                        sum += A[i][j] * x[j];
                    }
                }
                const newX = (b[i] - sum) / A[i][i];
                maxDiff = Math.max(maxDiff, Math.abs(newX - x[i]));
                x[i] = newX;
            }
            if (maxDiff < 1e-6) break;
        }
        return x;
    }

    static linearRegress(X, y) {
        const X1 = X.map(row => [...row, 1]);
        const coefficients = this.solveLinearSystem(X1, y);

        const coef = coefficients.slice(0, -1);
        const intercept = coefficients[coefficients.length - 1];
        const r2 = this.r2Score(y, this.predictLinear(X1, coefficients));

        return { coef, intercept, r2 };
    }

    static r2Score(yTrue, yPred) {
        const yMean = this.mean(yTrue);
        const tot = yTrue.reduce((sum, val, i) => sum + (val - yMean) * (val - yMean), 0);
        const res = yTrue.reduce((sum, val, i) => sum + (val - yPred[i]) * (val - yPred[i]), 0);

        return tot === 0 ? 0 : 1 - res / tot;
    }

    static predictLinear(X, coefficients) {
        return X.map(row =>
            row.reduce((sum, val, i) => sum + val * coefficients[i], 0)
        );
    }

    static scale(x) {
        if (!Array.isArray(x[0])) {
            const mean = this.mean(x);
            const std = this.std(x);
            return x.map(val => (val - mean) / (std || 1));
        }

        const means = this.mean(x, 0);
        const stds = x[0].map((_, i) => {
            const col = x.map(row => row[i]);
            return this.std(col) || 1;
        });

        return x.map(row =>
            row.map((val, i) => (val - means[i]) / stds[i])
        );
    }

    static listToDummy(blocks) {
        const nLv = blocks.length;
        const nMv = blocks.reduce((sum, block) => sum + block.length, 0);
        const wMat = Array(nMv).fill().map(() => Array(nLv).fill(0));
        const inds = this.indexify(blocks);

        inds.forEach((y, x) => {
            wMat[x][y] = 1;
        });

        return wMat;
    }

    static indexify(blocks) {
        const inds = [];
        blocks.forEach((block, i) => {
            inds.push(...Array(block.length).fill(i));
        });
        return inds;
    }

    static getBlocks(ends) {
        const blocks = [];
        let start = 0;
        ends.forEach(end => {
            const block = [];
            for (let i = start; i < end; i++) {
                block.push(i);
            }
            blocks.push(block);
            start = end;
        });
        return blocks;
    }

    static evalGOF(communality, r2, blocks) {
        const comu = [...communality];
        let x = 0;

        blocks.forEach(blk => {
            const nBlk = blk.length;
            if (nBlk < 2) {
                comu[blk[0]] = 0.0;
            }
            x += nBlk;
        });

        const nnzMean = arr => {
            const nonZero = arr.filter(val => val !== 0);
            return nonZero.length > 0 ? nonZero.reduce((a, b) => a + b) / nonZero.length : 0;
        };

        return Math.sqrt(nnzMean(comu) * nnzMean(r2));
    }

    // الدوال الرئيسية للنموذج
    getLatentVariable(X) {
        const XManifest = this.calcManifest(X, this.blocks);
        const XScaled = XManifest.map(row =>
            row.map((val, i) => (val - this.xMean[i]) / this.xStd[i])
        );
        return this.matrixMultiply(XScaled, this.outerWeight);
    }

    predict(X) {
        const latentVariable = this.getLatentVariable(X);
        const lastLV = latentVariable[0].length - 1;
        return latentVariable.map(row => {
            const score = row.reduce((sum, val, i) =>
                sum + val * this.pathCoef[lastLV][i], 0
            ) + this.pathIntercept[lastLV];
            return score;
        });
    }

    fit(X, pathMatrix, blocks, modes) {
        // معالجة البيانات الأولية
        this.pathMatrix = pathMatrix.map(row => [...row]);
        this.blocks = blocks.map(block => [...block]);
        const outerMat = this.constructor.listToDummy(blocks);
        this.outerMatrix = outerMat.map(row => [...row]);
        this.modes = [...modes];

        const XManifest = this.calcManifest(X, blocks);
        this.xStd = this.std(XManifest, 0).map(val => val || 1);
        this.xMean = this.mean(XManifest, 0);
        const XScaled = this.constructor.scale(XManifest);

        // تركيب النموذج الخارجي
        const wMat = this.calcWeightOuter(XScaled, pathMatrix, blocks, modes);
        this.outerWeight = wMat.map(row => [...row]);
        const latentVariable = this.calcLatentVariable(XScaled, wMat, outerMat);
        this.scores = latentVariable.map(row => [...row]);

        // تركيب النموذج الداخلي
        const pathRet = this.calcWeightInner(pathMatrix, latentVariable);
        this.pathCoef = pathRet.coef.map(row => [...row]);
        this.pathIntercept = [...pathRet.intercept];
        const effects = this.calcEffects(this.pathCoef);
        this.indirectEffect = effects.indirect;
        this.totalEffect = effects.total;
        this.pathR2 = [...pathRet.r2];

        // التقييم
        if (this.eval) {
            this.evaluateModel(XScaled, latentVariable, outerMat, blocks, wMat);
        }

        return this;
    }

    fitFromDefinitions(data, lvDefs, relations) {
        this.lvDefs = [...lvDefs];
        this.relations = [...relations];
        this.lvNames = lvDefs.map(lv => lv.name);

        // بناء مصفوفة المسار
        const lvNames = lvDefs.map(lv => lv.name);
        const lvIndex = {};
        lvNames.forEach((name, idx) => lvIndex[name] = idx);

        const pathMatrix = Array(lvNames.length).fill().map(() =>
            Array(lvNames.length).fill(0)
        );

        relations.forEach(rel => {
            const src = lvIndex[rel.source];
            const tgt = lvIndex[rel.target];
            pathMatrix[tgt][src] = 1;
        });

        // بناء الكتل والمؤشرات
        const blocks = [];
        const allIndicators = [];

        lvDefs.forEach(lv => {
            const indsIdx = lv.indicators.map(ind =>
                data.columns ? data.columns.indexOf(ind) : data[0].indexOf(ind)
            );
            blocks.push(indsIdx);
            allIndicators.push(...lv.indicators);
        });

        const ends = [];
        let count = 0;
        lvDefs.forEach(lv => {
            count += lv.indicators.length;
            ends.push(count);
        });

        const getBlocks = this.constructor.getBlocks(ends);
        const modes = lvDefs.map(lv => lv.mode);

        // تحضير بيانات الإدخال
        let X;
        if (data.columns) {
            // DataFrame-like object
            X = allIndicators.map(col => data[col]);
            X = this.transpose(X);
        } else {
            // Array of arrays
            X = data.map(row => allIndicators.map(ind => row[ind]));
        }

        return this.fit(X, pathMatrix, getBlocks, modes);
    }

    evaluateModel(X, latentVariable, outerMat, blocks, wMat) {
        const xloads = this.constructor.correlate(X, latentVariable);
        const loadings = this.extractLoadings(xloads, outerMat);

        this.communality = loadings.map(val => val * val);
        const r2 = this.matrixVectorMultiply(outerMat, this.pathR2);
        this.redundancy = this.communality.map((val, i) => val * r2[i]);
        this.gof = this.constructor.evalGOF(this.communality, this.pathR2, blocks);

        this.xloads = xloads.map(row => [...row]);
        this.loadings = [...loadings];

        if (this.plsc) {
            this.plscLoadings = Metrics.applyPLSCCorrection(X, this.scores, this.outerMatrix);
        }

        // حساب المقاييس الإضافية
        this.ave = Metrics.computeAVE(this.loadings, blocks);
        this.cr = Metrics.computeCR(this.loadings, blocks);
        this.rhoA = Metrics.computeRhoA(wMat, X, blocks);
        this.alpha = Metrics.computeAlpha(X, blocks);
        this.htmt = Metrics.computeHTMT(X, blocks);
        this.fornellLarcker = Metrics.computeFornellLarcker(this.scores, this.ave);
    }

    // دوال حسابية مساعدة
    calcManifest(X, blocks) {
        const indBlock = blocks.flat();
        return X.map(row => indBlock.map(idx => row[idx]));
    }

    calcLatentVariable(X, wMat, outerMat) {
        const latentVariable = this.matrixMultiply(X, wMat);
        const covXY = this.matrixMultiply(this.transpose(X), latentVariable);

        const wSign = covXY.map((row, i) =>
            row.map((val, j) => Math.sign(val * outerMat[i][j]))
        );

        const signSum = wSign[0].map((_, j) =>
            wSign.reduce((sum, row) => sum + row[j], 0)
        );

        const finalSign = signSum.map(val => {
            const sgn = Math.sign(val);
            return sgn === 0 ? -1 : sgn;
        });

        return latentVariable.map(row =>
            row.map((val, j) => val * finalSign[j])
        );
    }

    calcWeightOuter(X, pathMatrix, blocks, modes, scheme = "path", maxIter = 100, tol = 1e-6) {
        const nSamples = X.length;
        const nMv = X[0].length;
        const nLv = pathMatrix.length;

        const blockInds = this.constructor.indexify(blocks);
        const outerMat = this.constructor.listToDummy(blocks);

        let wStd = this.std(this.matrixMultiply(X, outerMat), 0);
        wStd = wStd.map(val => val || 1);

        let wMat = outerMat.map((row, i) =>
            row.map((val, j) => val / wStd[j])
        );

        let wOld = wMat.map(row => row.reduce((a, b) => a + b, 0));
        let finalIter = maxIter;

        for (let iter = 0; iter < maxIter; iter++) {
            // تقدير خارجي للمتغيرات الكامنة
            let Y = this.matrixMultiply(X, wMat);
            Y = this.constructor.scale(Y);

            // مصفوفة الأوزان الداخلية
            let E;
            if (scheme === "centroid") {
                const corr = Metrics.computeCorrelation(Y);
                E = corr.map((row, i) =>
                    row.map((val, j) =>
                        Math.sign(val) * (pathMatrix[i][j] + pathMatrix[j][i])
                    )
                );
            } else if (scheme === "factorial") {
                const corr = Metrics.computeCorrelation(Y);
                E = corr.map((row, i) =>
                    row.map((val, j) => val * (pathMatrix[i][j] + pathMatrix[j][i]))
                );
            } else if (scheme === "path") {
                E = this.calcWeightPathScheme(pathMatrix, Y);
            } else {
                const corr = Metrics.computeCorrelation(Y);
                E = corr.map((row, i) =>
                    row.map((val, j) =>
                        Math.sign(val) * (pathMatrix[i][j] + pathMatrix[j][i])
                    )
                );
            }

            // تقدير داخلي للمتغيرات الكامنة
            let Z = this.matrixMultiply(Y, E);
            const zStd = this.std(Z, 0).map(val => val || 1);
            Z = Z.map(row => row.map((val, j) => val / zStd[j]));

            // حساب الأوزان الخارجية
            for (let j = 0; j < nLv; j++) {
                const inds = blockInds.map((val, idx) => val === j ? idx : -1)
                    .filter(idx => idx !== -1);

                const Xj = X.map(row => inds.map(idx => row[idx]));
                const zj = Z.map(row => row[j]);

                if (modes[j] === "A") {
                    const newWeights = this.constructor.correlate(Xj, zj);
                    inds.forEach((idx, k) => {
                        wMat[idx][j] = newWeights[k];
                    });
                } else if (modes[j] === "B") {
                    const newWeights = this.constructor.solveLinearSystem(Xj, zj);
                    inds.forEach((idx, k) => {
                        wMat[idx][j] = newWeights[k];
                    });
                } else {
                    const newWeights = this.constructor.correlate(Xj, zj);
                    inds.forEach((idx, k) => {
                        wMat[idx][j] = newWeights[k];
                    });
                }
            }

            const wNew = wMat.map(row => row.reduce((a, b) => a + b, 0));
            const wDif = wOld.reduce((sum, val, i) =>
                sum + Math.pow(Math.abs(val) - Math.abs(wNew[i]), 2), 0
            );

            if (wDif < tol) {
                finalIter = iter;
                break;
            }
            wOld = wNew;
        }

        console.log("Iteration:", finalIter);
        console.log("Tolerance:", wDif);

        wStd = this.std(this.matrixMultiply(X, wMat), 0);
        return wMat.map(row => row.map((val, j) => val / (wStd[j] || 1)));
    }

    calcWeightInner(pathMatrix, latentVariable) {
        const nRow = pathMatrix.length;
        const pathCoef = pathMatrix.map(row => row.map(val => val * 1.0));
        const pathIntercept = new Array(nRow).fill(0);
        const pathR2 = new Array(nRow).fill(0);

        const endogenous = pathMatrix.map(row => row.reduce((a, b) => a + b, 0) > 0);
        const indEndo = endogenous.map((val, i) => val ? i : -1).filter(i => i !== -1);

        indEndo.forEach(indDep => {
            const lvDep = latentVariable.map(row => row[indDep]);
            const indIndep = pathMatrix[indDep].map(val => val === 1);
            const lvIndep = latentVariable.map(row =>
                row.filter((_, i) => indIndep[i])
            );

            const lm = this.constructor.linearRegress(lvIndep, lvDep);

            let coefIndex = 0;
            pathMatrix[indDep].forEach((val, i) => {
                if (val === 1) {
                    pathCoef[indDep][i] = lm.coef[coefIndex++];
                }
            });

            pathIntercept[indDep] = lm.intercept;
            pathR2[indDep] = lm.r2;
        });

        return { coef: pathCoef, intercept: pathIntercept, r2: pathR2 };
    }

    calcWeightPathScheme(pathMatrix, latentVariable) {
        const pathWeight = pathMatrix.map(row => row.map(val => val * 1.0));
        const nSamples = latentVariable.length;
        const nLv = latentVariable[0].length;

        for (let k = 0; k < nLv; k++) {
            const yk = latentVariable.map(row => row[k]);

            // followers
            const follow = pathMatrix[k].map(val => val === 1);
            if (follow.some(val => val)) {
                const Xk = latentVariable.map(row =>
                    row.filter((_, i) => follow[i])
                );
                const weights = this.constructor.solveLinearSystem(Xk, yk);

                let weightIndex = 0;
                pathMatrix[k].forEach((val, i) => {
                    if (val === 1) {
                        pathWeight[i][k] = weights[weightIndex++];
                    }
                });
            }

            // predecessors
            const predec = pathMatrix.map(row => row[k] === 1);
            if (predec.some(val => val)) {
                const Xk = latentVariable.map(row =>
                    row.filter((_, i) => predec[i])
                );
                const weights = this.matrixVectorMultiply(this.transpose(Xk), yk);

                let weightIndex = 0;
                pathMatrix.forEach((row, i) => {
                    if (row[k] === 1) {
                        pathWeight[i][k] = weights[weightIndex++];
                    }
                });
            }
        }

        return pathWeight;
    }

    calcEffects(pathCoef) {
        const nLv = pathCoef.length;
        const directEffect = pathCoef.map(row => [...row]);
        let totalEffect = pathCoef.map(row => [...row]);
        let tmpEffect = pathCoef.map(row => [...row]);

        for (let k = 1; k < nLv - 1; k++) {
            tmpEffect = this.matrixMultiply(tmpEffect, pathCoef);
            totalEffect = totalEffect.map((row, i) =>
                row.map((val, j) => val + tmpEffect[i][j])
            );
        }

        const indirectEffect = totalEffect.map((row, i) =>
            row.map((val, j) => val - directEffect[i][j])
        );

        return { indirect: indirectEffect, total: totalEffect };
    }

    // دوال مساعدة للرياضيات
    static mean(array, axis = null) {
        if (axis === 0) {
            return array[0].map((_, i) =>
                array.reduce((sum, row) => sum + row[i], 0) / array.length
            );
        }
        return array.reduce((a, b) => a + b, 0) / array.length;
    }

    static std(array, axis = null) {
        if (axis === 0) {
            return array[0].map((_, i) => {
                const col = array.map(row => row[i]);
                const mean = this.mean(col);
                const variance = col.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / col.length;
                return Math.sqrt(variance);
            });
        }
        const mean = this.mean(array);
        const variance = array.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / array.length;
        return Math.sqrt(variance);
    }

    static transpose(matrix) {
        return matrix[0].map((_, i) => matrix.map(row => row[i]));
    }

    static multiplyMatrices(A, B) {
        return A.map(row =>
            B[0].map((_, j) =>
                row.reduce((sum, val, i) => sum + val * B[i][j], 0)
            )
        );
    }

    static multiplyMatrixVector(A, b) {
        return A.map(row =>
            row.reduce((sum, val, i) => sum + val * b[i], 0)
        );
    }

    matrixMultiply(A, B) {
        return this.constructor.multiplyMatrices(A, B);
    }

    matrixVectorMultiply(A, b) {
        return this.constructor.multiplyMatrixVector(A, b);
    }

    extractLoadings(xloads, outerMat) {
        const loadings = [];
        outerMat.forEach((row, i) => {
            row.forEach((val, j) => {
                if (val === 1) {
                    loadings.push(xloads[i]?.[j] || xloads[j]);
                }
            });
        });
        return loadings;
    }
}

// دوال إضافية لتحليل التأثير
function interpretF2(f2) {
    if (f2 < 0.02) return "لا يوجد تأثير تقريبًا";
    if (f2 < 0.15) return "تأثير صغير";
    if (f2 < 0.35) return "تأثير متوسط";
    return "تأثير كبير";
}

// تصدير الكلاس للاستخدام
// في نهاية ملف plspm.js تأكد من وجود هذا:
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PLSPM, interpretF2 };
} else {
    // للتشغيل في المتصفح مباشرة
    window.PLSPM = PLSPM;
    window.interpretF2 = interpretF2;
}