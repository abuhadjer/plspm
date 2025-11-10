// metrics.js - دوال تحليل PLS-PM مترجمة إلى JavaScript

class Metrics {
    /**
     * حساب متوسط التباين المستخرج (AVE)
     * @param {number[]} loadings - مصفوفة الأحمال
     * @param {number[][]} blocks - كتل المؤشرات لكل متغير كامن
     * @returns {number[]} قيم AVE لكل متغير كامن
     */
    static computeAVE(loadings, blocks) {
        const ave = [];
        let idx = 0;

        for (const block of blocks) {
            const blockLoadings = loadings.slice(idx, idx + block.length);
            const squaredLoadings = blockLoadings.map(x => x * x);
            const meanSquared = squaredLoadings.reduce((sum, val) => sum + val, 0) / squaredLoadings.length;
            ave.push(meanSquared);
            idx += block.length;
        }

        return ave;
    }

    /**
     * حساب معامل الموثوقية (CR)
     * @param {number[]} loadings - مصفوفة الأحمال
     * @param {number[][]} blocks - كتل المؤشرات لكل متغير كامن
     * @returns {number[]} قيم CR لكل متغير كامن
     */
    static computeCR(loadings, blocks) {
        const cr = [];
        let idx = 0;

        for (const block of blocks) {
            const l = loadings.slice(idx, idx + block.length);
            const sumL = l.reduce((sum, val) => sum + val, 0);
            const numerator = sumL * sumL;

            const sumSquared = l.reduce((sum, val) => sum + (1 - val * val), 0);
            const denominator = numerator + sumSquared;

            cr.push(numerator / denominator);
            idx += block.length;
        }

        return cr;
    }

    /**
     * حساب مصفوفة التغاير
     * @param {number[][]} matrix - مصفوفة البيانات
     * @returns {number[][]} مصفوفة التغاير
     */
    static computeCovariance(matrix) {
        const n = matrix.length;
        const means = matrix[0].map((_, colIndex) =>
            matrix.reduce((sum, row) => sum + row[colIndex], 0) / n
        );

        const cov = [];
        for (let i = 0; i < matrix[0].length; i++) {
            const row = [];
            for (let j = 0; j < matrix[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < n; k++) {
                    sum += (matrix[k][i] - means[i]) * (matrix[k][j] - means[j]);
                }
                row.push(sum / (n - 1));
            }
            cov.push(row);
        }

        return cov;
    }

    /**
     * حساب rhoA لجميع المتغيرات الكامنة
     * @param {number[][]} weights - مصفوفة الأوزان
     * @param {number[][]} X - بيانات المؤشرات
     * @param {number[][]} blocks - كتل المؤشرات
     * @returns {number[]} قيم rhoA لكل متغير كامن
     */
    static computeRhoA(weights, X, blocks) {
        const rhoAValues = [];

        for (let i = 0; i < blocks.length; i++) {
            const inds = blocks[i];

            if (inds.length === 1) {
                rhoAValues.push(1.0);
                continue;
            }

            // استخراج الأوزان للمتغير الكامن الحالي
            const w = inds.map(ind => [weights[ind][i]]);

            // استخراج بيانات الكتلة
            const XBlock = X.map(row => inds.map(ind => row[ind]));

            // حساب مصفوفة التغاير
            const S = this.computeCovariance(XBlock);

            // تعيين القطر الرئيسي إلى الصفر
            for (let j = 0; j < S.length; j++) {
                S[j][j] = 0;
            }

            // حساب AAnondiag = w * w.T
            const AAnondiag = [];
            for (let j = 0; j < w.length; j++) {
                const row = [];
                for (let k = 0; k < w.length; k++) {
                    row.push(w[j][0] * w[k][0]);
                }
                AAnondiag.push(row);
            }

            // تعيين القطر الرئيسي إلى الصفر
            for (let j = 0; j < AAnondiag.length; j++) {
                AAnondiag[j][j] = 0;
            }

            // حساب البسط: w.T * S * w
            let numerator = 0;
            for (let j = 0; j < w.length; j++) {
                for (let k = 0; k < w.length; k++) {
                    numerator += w[j][0] * S[j][k] * w[k][0];
                }
            }

            // حساب المقام: w.T * AAnondiag * w
            let denominator = 0;
            for (let j = 0; j < w.length; j++) {
                for (let k = 0; k < w.length; k++) {
                    denominator += w[j][0] * AAnondiag[j][k] * w[k][0];
                }
            }

            // حساب norm_squared
            let normSquared = 0;
            for (let j = 0; j < w.length; j++) {
                normSquared += w[j][0] * w[j][0];
            }
            normSquared = normSquared * normSquared;

            const rhoA = denominator !== 0 ? normSquared * (numerator / denominator) : NaN;
            rhoAValues.push(rhoA);
        }

        return rhoAValues;
    }

    /**
     * حساب معامل كرونباخ ألفا
     * @param {number[][]} X - بيانات المؤشرات
     * @param {number[][]} blocks - كتل المؤشرات
     * @returns {number[]} قيم ألفا لكل متغير كامن
     */
    static computeAlpha(X, blocks) {
        const alphas = [];

        for (const block of blocks) {
            const k = block.length;

            if (k <= 1) {
                alphas.push(1.0);
                continue;
            }

            // استخراج بيانات الكتلة
            const data = X.map(row => block.map(ind => row[ind]));

            // حساب تباينات الأعمدة
            const variances = [];
            for (let j = 0; j < k; j++) {
                const column = data.map(row => row[j]);
                const mean = column.reduce((sum, val) => sum + val, 0) / column.length;
                const variance = column.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (column.length - 1);
                variances.push(variance);
            }

            // حساب التباين الكلي للمجموع
            const rowSums = data.map(row => row.reduce((sum, val) => sum + val, 0));
            const rowSumsMean = rowSums.reduce((sum, val) => sum + val, 0) / rowSums.length;
            const totalVar = rowSums.reduce((sum, val) => sum + Math.pow(val - rowSumsMean, 2), 0) / (rowSums.length - 1);

            if (totalVar === 0) {
                alphas.push(NaN);
                continue;
            }

            const sumVariances = variances.reduce((sum, val) => sum + val, 0);
            const alpha = (k / (k - 1)) * (1 - sumVariances / totalVar);
            alphas.push(alpha);
        }

        return alphas;
    }

    /**
     * حساب مصفوفة الارتباط
     * @param {number[][]} matrix - مصفوفة البيانات
     * @returns {number[][]} مصفوفة الارتباط
     */
    static computeCorrelation(matrix) {
        const n = matrix.length;
        const p = matrix[0].length;

        // حساب المتوسطات
        const means = [];
        for (let j = 0; j < p; j++) {
            let sum = 0;
            for (let i = 0; i < n; i++) {
                sum += matrix[i][j];
            }
            means.push(sum / n);
        }

        // حساب الانحرافات المعيارية
        const stds = [];
        for (let j = 0; j < p; j++) {
            let sumSq = 0;
            for (let i = 0; i < n; i++) {
                sumSq += Math.pow(matrix[i][j] - means[j], 2);
            }
            stds.push(Math.sqrt(sumSq / (n - 1)));
        }

        // حساب مصفوفة الارتباط
        const corr = [];
        for (let i = 0; i < p; i++) {
            const row = [];
            for (let j = 0; j < p; j++) {
                let sum = 0;
                for (let k = 0; k < n; k++) {
                    sum += ((matrix[k][i] - means[i]) / stds[i]) * ((matrix[k][j] - means[j]) / stds[j]);
                }
                row.push(sum / (n - 1));
            }
            corr.push(row);
        }

        return corr;
    }

    /**
     * حساب مصفوفة HTMT
     * @param {number[][]} X - بيانات المؤشرات الموحدة
     * @param {number[][]} blocks - كتل المؤشرات
     * @returns {number[][]} مصفوفة HTMT
     */
    static computeHTMT(X, blocks) {
        const nLv = blocks.length;
        const htmtMatrix = Array(nLv).fill().map(() => Array(nLv).fill(0));
        const corrMatrix = this.computeCorrelation(X);

        for (let i = 0; i < nLv; i++) {
            for (let j = 0; j < nLv; j++) {
                if (i === j) {
                    htmtMatrix[i][j] = 1.0;
                } else if (i < j) {
                    const interCorrs = [];

                    // الارتباطات بين المجموعتين
                    for (const m of blocks[i]) {
                        for (const n of blocks[j]) {
                            interCorrs.push(Math.abs(corrMatrix[m][n]));
                        }
                    }

                    // الارتباطات داخل المجموعة الأولى
                    const intraCorrsI = [];
                    for (let a = 0; a < blocks[i].length; a++) {
                        for (let b = a + 1; b < blocks[i].length; b++) {
                            intraCorrsI.push(Math.abs(corrMatrix[blocks[i][a]][blocks[i][b]]));
                        }
                    }

                    // الارتباطات داخل المجموعة الثانية
                    const intraCorrsJ = [];
                    for (let a = 0; a < blocks[j].length; a++) {
                        for (let b = a + 1; b < blocks[j].length; b++) {
                            intraCorrsJ.push(Math.abs(corrMatrix[blocks[j][a]][blocks[j][b]]));
                        }
                    }

                    const allIntraCorrs = [...intraCorrsI, ...intraCorrsJ];
                    const denominator = allIntraCorrs.reduce((sum, val) => sum + val, 0) / allIntraCorrs.length;
                    const meanInter = interCorrs.reduce((sum, val) => sum + val, 0) / interCorrs.length;

                    htmtMatrix[i][j] = meanInter / denominator;
                    htmtMatrix[j][i] = htmtMatrix[i][j];
                }
            }
        }

        return htmtMatrix;
    }

    /**
     * حساب مصفوفة فورنل-لاركر
     * @param {number[][]} latentScores - درجات المتغيرات الكامنة
     * @param {number[]} ave - قيم AVE
     * @returns {number[][]} مصفوفة فورنل-لاركر
     */
    static computeFornellLarcker(latentScores, ave) {
        const corMatrix = this.computeCorrelation(latentScores);
        const flMatrix = corMatrix.map(row => [...row]);
        const sqrtAve = ave.map(val => Math.sqrt(val));

        for (let i = 0; i < flMatrix.length; i++) {
            flMatrix[i][i] = sqrtAve[i];
        }

        return flMatrix;
    }

    /**
     * تطبيق تصحيح PLSc على الأحمال الخارجية
     * @param {number[][]} X - بيانات المؤشرات الموحدة
     * @param {number[][]} latentScores - درجات المتغيرات الكامنة
     * @param {number[][]} outerMatrix - المصفوفة الخارجية الثنائية
     * @returns {number[]} الأحمال المصححة
     */
    static applyPLSCCorrection(X, latentScores, outerMatrix) {
        const nMv = outerMatrix.length;
        const nLv = outerMatrix[0].length;
        const correctedLoadings = Array(nMv).fill(0);

        for (let j = 0; j < nLv; j++) {
            const indicators = [];
            for (let i = 0; i < nMv; i++) {
                if (outerMatrix[i][j] === 1) {
                    indicators.push(i);
                }
            }

            for (const i of indicators) {
                const x_i = X.map(row => row[i]);
                const y_j = latentScores.map(row => row[j]);

                // حساب الارتباط
                const meanX = x_i.reduce((sum, val) => sum + val, 0) / x_i.length;
                const meanY = y_j.reduce((sum, val) => sum + val, 0) / y_j.length;

                const stdX = Math.sqrt(x_i.reduce((sum, val) => sum + Math.pow(val - meanX, 2), 0) / x_i.length);
                const stdY = Math.sqrt(y_j.reduce((sum, val) => sum + Math.pow(val - meanY, 2), 0) / y_j.length);

                let covariance = 0;
                for (let k = 0; k < x_i.length; k++) {
                    covariance += (x_i[k] - meanX) * (y_j[k] - meanY);
                }
                covariance /= x_i.length;

                const corr = covariance / (stdX * stdY);
                correctedLoadings[i] = corr;
            }
        }

        return correctedLoadings;
    }
}

// دوال مساعدة إضافية
class MathUtils {
    /**
     * حساب متوسط مصفوفة
     */
    static mean(array) {
        return array.reduce((sum, val) => sum + val, 0) / array.length;
    }

    /**
     * حساب الانحراف المعياري
     */
    static std(array, ddof = 0) {
        const mean = this.mean(array);
        const variance = array.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (array.length - ddof);
        return Math.sqrt(variance);
    }

    /**
     * توحيد البيانات (Z-scores)
     */
    static standardize(matrix) {
        return matrix.map(row => {
            const rowMean = this.mean(row);
            const rowStd = this.std(row);
            return row.map(val => (val - rowMean) / rowStd);
        });
    }
}


// في نهاية ملف metrics.js تأكد من وجود هذا:
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Metrics, MathUtils };
} else {
    // للتشغيل في المتصفح مباشرة
    window.Metrics = Metrics;
    window.MathUtils = MathUtils;
}