#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <algorithm>
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

static void die(const string& msg) { throw std::runtime_error(msg); }

// -------- CSV Reader (numeric only) --------
static Matrix readCsvMatrix(const string& path) {
    std::ifstream fin(path);
    if (!fin) die("Failed to open CSV: " + path);

    vector<vector<double>> rows;
    string line;

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        string cell;
        vector<double> row;

        while (std::getline(ss, cell, ',')) {
            // trim spaces
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);

            if (cell.empty()) continue;
            row.push_back(std::stod(cell));
        }

        if (!row.empty()) rows.push_back(std::move(row));
    }

    if (rows.empty()) die("CSV has no data: " + path);

    int r = (int)rows.size();
    int c = (int)rows[0].size();
    for (int i = 1; i < r; ++i) {
        if ((int)rows[i].size() != c) die("CSV column mismatch: " + path);
    }

    Matrix M(r, c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            M(i, j) = rows[i][j];

    return M;
}

static vector<double> readCsvVector(const string& path) {
    Matrix m = readCsvMatrix(path);
    if (m.c == 1) {
        vector<double> v(m.r);
        for (int i = 0; i < m.r; ++i) v[i] = m(i, 0);
        return v;
    }
    if (m.r == 1) {
        vector<double> v(m.c);
        for (int j = 0; j < m.c; ++j) v[j] = m(0, j);
        return v;
    }
    die("Vector CSV must be 1xN or Nx1: " + path);
    return {};
}

// -------- Math ops --------
static Matrix transpose(const Matrix& A) {
    Matrix T(A.c, A.r);
    for (int i = 0; i < A.r; ++i)
        for (int j = 0; j < A.c; ++j)
            T(j, i) = A(i, j);
    return T;
}

static Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.c != B.r) die("matmul dimension mismatch");
    Matrix C(A.r, B.c, 0.0);
    for (int i = 0; i < A.r; ++i) {
        for (int k = 0; k < A.c; ++k) {
            double aik = A(i, k);
            for (int j = 0; j < B.c; ++j) {
                C(i, j) += aik * B(k, j);
            }
        }
    }
    return C;
}

// A(r x c) + b(r) broadcast to all columns
static Matrix addBiasCol(const Matrix& A, const vector<double>& b) {
    if ((int)b.size() != A.r) die("addBiasCol bias size mismatch");
    Matrix C = A;
    for (int i = 0; i < A.r; ++i)
        for (int j = 0; j < A.c; ++j)
            C(i, j) += b[i];
    return C;
}

// bipolar sigmoid: 2/(1+exp(-2x)) - 1
static Matrix bipolarSigmoid(const Matrix& X) {
    Matrix Y(X.r, X.c);
    for (int i = 0; i < X.r; ++i)
        for (int j = 0; j < X.c; ++j) {
            double x = X(i, j);
            Y(i, j) = 2.0 / (1.0 + std::exp(-2.0 * x)) - 1.0;
        }
    return Y;
}

static double normalize_bipolar(double x, double mn, double mx) {
    double denom = (mx - mn);
    if (std::abs(denom) < 1e-12) return 0.0;
    return -1.0 + 2.0 * (x - mn) / denom;
}

static void writePredictionCsv(const string& path,
                               const vector<double>& rawO,
                               const vector<double>& newO,
                               const vector<double>& Ycm) {
    std::ofstream fout(path);
    if (!fout) die("Failed to write: " + path);
    fout << "index,O_raw,new_O,Y_cm\n";
    for (size_t i = 0; i < rawO.size(); ++i) {
        fout << (i + 1) << ","
             << std::setprecision(12) << rawO[i] << ","
             << std::setprecision(12) << newO[i] << ","
             << std::setprecision(12) << Ycm[i] << "\n";
    }
}

int main() {
    try {
        // ====== Files (CSV) ======
        const string INPUTS_CSV   = "ANNSFM_inputs.csv";
        const string CONFIGI_CSV  = "ANNSFM_Config_I.csv";
        const string HW_CSV       = "ANNSFM_CS_HW.csv";
        const string HB_CSV       = "ANNSFM_CS_HB.csv";
        const string OW_CSV       = "ANNSFM_CS_OW.csv";
        const string OB_CSV       = "ANNSFM_CS_OB.csv";

        // ====== Load ======
        Matrix inputs = readCsvMatrix(INPUTS_CSV);   // N x F
        Matrix cfgI   = readCsvMatrix(CONFIGI_CSV);  // 2 x F
        Matrix HW     = readCsvMatrix(HW_CSV);       // HN x F
        vector<double> HB = readCsvVector(HB_CSV);   // HN
        Matrix OW     = readCsvMatrix(OW_CSV);       // OUT x HN
        vector<double> OB = readCsvVector(OB_CSV);   // OUT

        int N = inputs.r;
        int F = inputs.c;
        if (cfgI.r != 2 || cfgI.c != F) die("Config_I must be 2 x F");
        if (HW.c != F) die("CS_HW must be HN x F");
        int HN = HW.r;
        if ((int)HB.size() != HN) die("CS_HB must be HN x 1");
        if (OW.c != HN) die("CS_OW must be OUT x HN");
        int OUT = OW.r;
        if ((int)OB.size() != OUT) die("CS_OB must be OUT x 1");

        // ====== Normalize inputs to [-1,1] ======
        vector<double> minI(F), maxI(F);
        for (int j = 0; j < F; ++j) {
            minI[j] = cfgI(0, j);
            maxI[j] = cfgI(1, j);
        }

        Matrix Xn(N, F);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < F; ++j)
                Xn(i, j) = normalize_bipolar(inputs(i, j), minI[j], maxI[j]);

        // MATLAB: W = CS_HW * inputs_normalization'
        Matrix Xnt = transpose(Xn);           // F x N
        Matrix W   = matmul(HW, Xnt);         // HN x N
        Matrix H   = addBiasCol(W, HB);       // HN x N
        Matrix HE  = bipolarSigmoid(H);       // HN x N

        // MATLAB: O = CS_OW * H_E + CS_OB
        Matrix Omat = matmul(OW, HE);         // OUT x N
        Omat = addBiasCol(Omat, OB);          // OUT x N

        // ====== If OUT>=1, use the first output for plotting/export (like MATLAB) ======
        vector<double> O_raw(N);
        for (int i = 0; i < N; ++i) O_raw[i] = Omat(0, i);

        double max_O = *std::max_element(O_raw.begin(), O_raw.end());
        double min_O = *std::min_element(O_raw.begin(), O_raw.end());

        vector<double> new_O(N), Y_cm(N);
        for (int i = 0; i < N; ++i) {
            // MATLAB: new_O = -1+ 2*(O-min_O)/(max_O-min_O)
            new_O[i] = normalize_bipolar(O_raw[i], min_O, max_O);

            // MATLAB: Y = (((O+1)/2*(max_O-min_O)+min_O)+0.7)*100
            double Y = (((O_raw[i] + 1.0) / 2.0 * (max_O - min_O) + min_O) + 0.7) * 100.0;
            Y_cm[i] = Y;
        }

        writePredictionCsv("prediction_validate.csv", O_raw, new_O, Y_cm);

        std::cout << "[OK] N=" << N << ", F=" << F << ", HN=" << HN << ", OUT=" << OUT << "\n";
        std::cout << "[OK] Saved: prediction_validate.csv\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}