#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iomanip>

using std::vector;
using std::string;

struct Matrix {
    int r = 0, c = 0;
    vector<double> a; // row-major

    Matrix() = default;
    Matrix(int rows, int cols, double v = 0.0) : r(rows), c(cols), a(rows * cols, v) {}

    double& operator()(int i, int j) { return a[i * c + j]; }
    double  operator()(int i, int j) const { return a[i * c + j]; }
};

static void die(const string& msg) {
    throw std::runtime_error(msg);
}

// 讀取「純數字」txt（以空白/換行分隔）
static vector<double> readAllNumbers(const string& path) {
    std::ifstream fin(path);
    if (!fin) die("Failed to open file: " + path);

    vector<double> nums;
    double x;
    while (fin >> x) nums.push_back(x);
    if (nums.empty()) die("No numeric data found in: " + path);
    return nums;
}

// 讀固定尺寸矩陣
static Matrix readMatrixFixed(const string& path, int rows, int cols) {
    auto nums = readAllNumbers(path);
    if ((int)nums.size() != rows * cols) {
        die("Size mismatch in " + path + " (expected " + std::to_string(rows * cols) +
            ", got " + std::to_string(nums.size()) + ")");
    }
    Matrix m(rows, cols);
    m.a = std::move(nums);
    return m;
}

// 讀「向量」(N x 1)
static vector<double> readVectorFixed(const string& path, int n) {
    auto nums = readAllNumbers(path);
    if ((int)nums.size() != n) {
        die("Size mismatch in " + path + " (expected " + std::to_string(n) +
            ", got " + std::to_string(nums.size()) + ")");
    }
    return nums;
}

// 矩陣乘法：A(r x k) * B(k x c) = C(r x c)
static Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.c != B.r) die("matmul dimension mismatch");
    Matrix C(A.r, B.c, 0.0);
    for (int i = 0; i < A.r; ++i) {
        for (int k = 0; k < A.c; ++k) {
            const double aik = A(i, k);
            for (int j = 0; j < B.c; ++j) {
                C(i, j) += aik * B(k, j);
            }
        }
    }
    return C;
}

// C = A + b （b 為 r x 1，會廣播到每一欄）
static Matrix addBiasCol(const Matrix& A, const vector<double>& b) {
    if ((int)b.size() != A.r) die("addBiasCol: bias size mismatch");
    Matrix C = A;
    for (int i = 0; i < A.r; ++i) {
        for (int j = 0; j < A.c; ++j) {
            C(i, j) += b[i];
        }
    }
    return C;
}

// H_E = 2/(1+exp(-2H)) - 1  （等價 tanh）
static Matrix bipolarSigmoid(const Matrix& H) {
    Matrix Y(H.r, H.c);
    for (int i = 0; i < H.r; ++i) {
        for (int j = 0; j < H.c; ++j) {
            const double x = H(i, j);
            Y(i, j) = 2.0 / (1.0 + std::exp(-2.0 * x)) - 1.0;
        }
    }
    return Y;
}

// Normalize to [-1, 1] using (x - min)/(max-min)
static double normalize_bipolar(double x, double mn, double mx) {
    const double denom = (mx - mn);
    if (std::abs(denom) < 1e-12) return 0.0; // avoid div0
    return -1.0 + 2.0 * (x - mn) / denom;
}

static void saveCsvVector(const string& path, const vector<double>& y) {
    std::ofstream fout(path);
    if (!fout) die("Failed to write: " + path);
    fout << "index,pred\n";
    for (size_t i = 0; i < y.size(); ++i) {
        fout << (i + 1) << "," << std::setprecision(12) << y[i] << "\n";
    }
}

int main() {
    try {
        // ====== 你原本程式固定的資料規格（依你 MATLAB / txt 用法）======
        const int N = 708;  // samples (data points)
        const int F = 8;    // input features
        const int HN = 12;  // hidden neurons

        // ====== 檔案名稱（沿用你原始專題命名）======
        const string INPUTS_TXT   = "ANNSFM_inputs.txt";        // N x F
        const string CONFIG_I_TXT = "ANNSFM_Config_I.txt";      // 2 x F (row0=min, row1=max)

        const string HW_TXT = "ANNSFM_CS_HW.txt";               // HN x F
        const string HB_TXT = "ANNSFM_CS_HB.txt";               // HN
        const string OW_TXT = "ANNSFM_CS_OW.txt";               // (out x HN) 可能是 1xHN 或 NxHN
        const string OB_TXT = "ANNSFM_CS_OB.txt";               // out

        // ====== 讀取 inputs 與 config ======
        Matrix inputs = readMatrixFixed(INPUTS_TXT, N, F);
        Matrix cfgI   = readMatrixFixed(CONFIG_I_TXT, 2, F);

        vector<double> inMin(F), inMax(F);
        for (int j = 0; j < F; ++j) {
            inMin[j] = cfgI(0, j);
            inMax[j] = cfgI(1, j);
        }

        // ====== inputs normalization to [-1, 1] ======
        Matrix Xn(N, F);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < F; ++j) {
                Xn(i, j) = normalize_bipolar(inputs(i, j), inMin[j], inMax[j]);
            }
        }

        // ====== 轉置成 (F x N) 以符合 MATLAB: CS_HW * inputs_normalization' ======
        Matrix Xnt(F, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < F; ++j)
                Xnt(j, i) = Xn(i, j);

        // ====== Hidden layer: H = HW * Xnt + HB ======
        Matrix HW = readMatrixFixed(HW_TXT, HN, F);
        vector<double> HB = readVectorFixed(HB_TXT, HN);

        Matrix W = matmul(HW, Xnt);         // HN x N
        Matrix H = addBiasCol(W, HB);       // HN x N
        Matrix HE = bipolarSigmoid(H);      // HN x N

        // ====== Output layer: O = OW * HE + OB ======
        // OW/OB 的維度你原始 C 與 MATLAB 有衝突，因此這裡做「自動判斷」：
        // - 若 OW 只有 12 個值 => 視為 (1 x HN)，輸出 O 為 (1 x N)
        // - 若 OW 有 k*HN 個值 => 視為 (k x HN)，輸出 O 為 (k x N)
        // OB 需對應 k

        auto owNums = readAllNumbers(OW_TXT);
        if ((int)owNums.size() % HN != 0) die("CS_OW size must be multiple of hidden size(12).");
        int OUT = (int)owNums.size() / HN;

        Matrix OW(OUT, HN);
        OW.a = std::move(owNums);

        vector<double> OB = readVectorFixed(OB_TXT, OUT);

        Matrix Omat = matmul(OW, HE);      // OUT x N
        Omat = addBiasCol(Omat, OB);       // OUT x N

        // ====== 若 OUT==1：輸出 708 點預測；否則輸出 OUT 組預測（都存 CSV）======
        // 你 MATLAB 後面有做 new_O 及 Y 的轉換，這裡保留「輸出 raw O」讓你比對。
        if (OUT == 1) {
            vector<double> y(N);
            for (int i = 0; i < N; ++i) y[i] = Omat(0, i);
            saveCsvVector("prediction.csv", y);
            std::cout << "[OK] Saved: prediction.csv (N=" << N << ")\n";
        } else {
            // 多輸出情況：存成 prediction_matrix.csv
            std::ofstream fout("prediction_matrix.csv");
            if (!fout) die("Failed to write prediction_matrix.csv");
            // header
            fout << "out_index";
            for (int i = 0; i < N; ++i) fout << ",t" << (i + 1);
            fout << "\n";
            for (int o = 0; o < OUT; ++o) {
                fout << (o + 1);
                for (int i = 0; i < N; ++i) fout << "," << std::setprecision(12) << Omat(o, i);
                fout << "\n";
            }
            std::cout << "[OK] Saved: prediction_matrix.csv (OUT=" << OUT << ", N=" << N << ")\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}