#ifndef MMATRIX_H // the 'include guard'
#define MMATRIX_H

#include <vector>

// Class that represents a mathematical matrix
class Matrix
{
public:
	// constructors
	Matrix() : nRows(0), nCols(0) {}
	Matrix(int n, int m, double x = 0) : nRows(n), nCols(m), A(n * m, x) {}
    
	// set all matrix entries equal to a double
	Matrix &operator=(double x)
	{
		for (unsigned i = 0; i < nRows * nCols; i++) A[i] = x;
		return *this;
	}

	// access element, indexed by (row, column) [rvalue]
	double operator()(int i, int j) const
	{
		return A[j + i * nCols];
	}

	// access element, indexed by (row, column) [lvalue]
	double &operator()(int i, int j)
	{
		return A[j + i * nCols];
	}
    
    friend ostream& operator<<(ostream& out, const Matrix& M);
    

	// size of matrix
	int Rows() const { return nRows; }
	int Cols() const { return nCols; }

private:
	unsigned int nRows, nCols;
	std::vector<double> A;
};

std::ostream &operator << (std::ostream &out, Matrix &M)
{
    int nRows {M.nRows};
    int nCols {M.nCols};
    
    for (int i{0}; i< nRows; i++)
    {
        out << "( ";
        for (int j{0}; j< nCols; j++) out << M(i,j) << " " ;
        out << ")" << '\n';
    }
    return out;
}

#endif
