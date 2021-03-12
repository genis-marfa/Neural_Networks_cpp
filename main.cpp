#include "mvector.h"
#include "mmatrix.h"
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <cassert>


////////////////////////////////////////////////////////////////////////////////
// Set up random number generation

// Set up a "random device" that generates a new random number each time the program is run
std::random_device rand_dev;

// Set up a pseudo-random number generater "rnd", seeded with a random number
std::mt19937 rnd(rand_dev());

// Alternative: set up the generator with an arbitrary constant integer. This can be useful for
// debugging because the program produces the same sequence of random numbers each time it runs.
// To get this behaviour, uncomment the line below and comment the declaration of "rnd" above.
//std::mt19937 rnd(12345);


////////////////////////////////////////////////////////////////////////////////
// Some operator overloads to allow arithmetic with MMatrix and MVector.
// These may be useful in helping write the equations for the neural network in
// vector form without having to loop over components manually.
//
// You may not need to use all of these; conversely, you may wish to add some
// more overloads.
double sech(double x){return 1.0/std::cosh(x);}

// MMatrix * MVector
MVector operator*(const MMatrix &m, const MVector &v)
{
    assert(m.Cols() == v.size());           // Ensure same dimensions for multiplication.
    
    MVector r(m.Rows());                    // Create vector r of zeros, of dimension equal to rows of m matrix

    for (int i=0; i<m.Rows(); i++)          // Loop over all matrix rows.
    {
        for (int j=0; j<m.Cols(); j++)      // Loop over all matrix columns.
        {
            r[i]+=m(i,j)*v[j];              // Multiply matrix row times vector entries and add them up, append to new vector r.
        }
    }
    return r;                               // Return result.
}

// transpose(MMatrix) * MVector
MVector TransposeTimes(const MMatrix &m, const MVector &v)
{
    assert(m.Rows() == v.size());           // Ensure same dimensions for multiplication.

    MVector r(m.Cols());                    // Create vector r of zeros, of dimension equal to columns of m matrix

    for (int i=0; i<m.Cols(); i++)          // Loop over all matrix columns.
    {
        for (int j=0; j<m.Rows(); j++)      // Loop over all matrix rows.
        {
            r[i]+=m(j,i)*v[j];              // Multiply matrix column times vector entries and add them up, append to new vector r.
        }
    }
    return r;                               // Return result.
}

// MVector + MVector
MVector operator+(const MVector &lhs, const MVector &rhs)
{
    assert(lhs.size() == rhs.size());       // Ensure vectors of same dimension for addition.

    MVector r(lhs);                         // Create vector r equal to one of the two vectors in addition.
    for (int i=0; i<lhs.size(); i++)
        r[i] += rhs[i];                     // Loop over vector components, add other vector and append to r.

    return r;                               // Return result.
}

// MVector - MVector
MVector operator-(const MVector &lhs, const MVector &rhs)
{
    assert(lhs.size() == rhs.size());       // Ensure vectors of same dimension for subtraction.

    MVector r(lhs);                         // Create vector r equal to one of the two vectors in subtraction.
    for (int i=0; i<lhs.size(); i++)
        r[i] -= rhs[i];                     // Loop over vector components, subtract other vector and append to r.

    return r;                               // Return result
}

// MMatrix = MVector <outer product> MVector.
// M = a <outer product> b
// Outter product: For 2 vectors u=[u_1,...,u_n], v=[v_1,..., v_m], outter product returns n x m matrix
// of entries equal to: | u_1v_1  u_1v_2 ... u_1v_m  |
//                      | u_2v_1  u_2v_2 ... u_2_v_m |
//                      |  ...      ...       ...    |
//                      | u_nv_1  u_nv_2 ... u_nv_m  |

MMatrix OuterProduct(const MVector &a, const MVector &b)
{
    MMatrix m(a.size(), b.size());      // Create a matrix of n_rows=size of vector a,  n_cols=size of vector b.
    for (int i=0; i<a.size(); i++)      // Loop over components of vector a
    {
        for (int j=0; j<b.size(); j++)  // Loop over components of vector b
        {
            m(i,j) = a[i]*b[j];         // Asign to the i,j th matrix entry, the product a_i b_j
        }
    }
    return m;                           // Return matrix.
}

// Hadamard product: Take two matrices (or vectors) of same dimension, do entry by entry multiplication:
// | a_1 a_2 |  * | b_1 b_2 | = | a_1b_1 a_2b_2 |
// | a_3 a_4 |    | b_3 b_4 |   | a_3b_3 a_4b_4 |
MVector operator*(const MVector &a, const MVector &b) // For vectors, this is just the dot product.
{
    assert(a.size() == b.size());     // Ensure vectors of same size
    
    MVector r(a.size());              // Create vector r of zeros, of dimension equal to vector's sizes.
    for (int i=0; i<a.size(); i++)    // Loop over vector entries.
        r[i]=a[i]*b[i];               // Asign to components of r, the product of vector entries.
    return r;
}

// double * MMatrix: Matrix multiplication by scalar.
MMatrix operator*(double d, const MMatrix &m)
{
    MMatrix r(m);                       // Create matrix r equal to matrix m.
    for (int i=0; i<m.Rows(); i++)      // Loop over matrix rows
        for (int j=0; j<m.Cols(); j++)  // Loop over matrix columns.
            r(i,j)*=d;                  // Multiply entries by scalar d.

    return r;                           // Return result.
}

// double * MVector: Vector multiplication by scalar. Similar to above.
MVector operator*(double d, const MVector &v)
{
    MVector r(v);
    for (int i=0; i<v.size(); i++)
        r[i]*=d;

    return r;
}

// MVector -= MVector: Returns output of subtracting a vector from another vector.
MVector operator-=(MVector &v1, const MVector &v) // v1-=v is equivalent to v1=v1-v.
{
    assert(v1.size()==v.size());        // Ensure vector sizes match.
    
    MVector r(v1);                      // Create vector r, equal to first vector v_1
    for (int i=0; i<v1.size(); i++)     // Loop over vector components.
        v1[i]-=v[i];                    // Subtract second vector's entries from v_1

    return r;                           // Return result.
}

// MMatrix -= MMatrix: Similar to above for matrices. m1 -= m2 is equivalent to m1 = m1 - m2
MMatrix operator-=(MMatrix &m1, const MMatrix &m2)
{
    assert (m1.Rows() == m2.Rows() && m1.Cols() == m2.Cols());

    for (int i=0; i<m1.Rows(); i++)
        for (int j=0; j<m1.Cols(); j++)
            m1(i,j)-=m2(i,j);

    return m1;
}

// Overload insertion operator: Display all components of vector in format (v1, v2, ... , vn)
inline std::ostream &operator<<(std::ostream &os, const MVector &rhs)
{
    std::size_t n = rhs.size();
    os << "(";
    for (std::size_t i=0; i<n; i++)
    {
        os << rhs[i];
        if (i!=(n-1)) os << ", ";
    }
    os << ")";
    return os;
}

// Output function for MMatrix:
// Display all components of matrix in format: (v11 v12 ... v1n )
//                                             (v21 v22 ... v2n )
//                                             (... ...     ... )
//                                             (vm1 vm2 ... vmn )
inline std::ostream &operator<<(std::ostream &os, const MMatrix &a)
{
    int c = a.Cols(), r = a.Rows();
    for (int i=0; i<r; i++)
    {
        os<<"(";
        for (int j=0; j<c; j++)
        {
            os.width(10);
            os << a(i,j);
            os << ((j==c-1)?')':',');
        }
        os << "\n";
    }
    return os;
}

////////////////////////////////////////////////////////////////////////////////
// Functions that provide sets of training data

// Generate 16 points of training data in the pattern illustrated in the project description

// Test data used to train netword, and desired outputs.
void GetTestData(std::vector<MVector> &x, std::vector<MVector> &y)
{
    x = {{0.125,.175}, {0.375,0.3125}, {0.05,0.675}, {0.3,0.025}, {0.15,0.3}, {0.25,0.5}, {0.2,0.95}, {0.15, 0.85},
         {0.75, 0.5}, {0.95, 0.075}, {0.4875, 0.2}, {0.725,0.25}, {0.9,0.875}, {0.5,0.8}, {0.25,0.75}, {0.5,0.5}};
    
    y = {{1},{1},{1},{1},{1},{1},{1},{1},
         {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1}};
}

// Generate 1000 points of test data in a checkerboard pattern
void GetCheckerboardData(std::vector<MVector> &x, std::vector<MVector> &y)
{
    std::mt19937 lr;                                // Generate random number lr
    x = std::vector<MVector>(1000, MVector(2));     // x: Vector of (2D vectors of) test data.
    y = std::vector<MVector>(1000, MVector(1));     // y: Vector of (1D vectors of) test data outputs.

    for (int i=0; i<1000; i++)                      // Asign these vectors values.
    {
        // Asign to ith entry of x, a vector {random num, random num}:
        x[i]={lr()/static_cast<double>(lr.max()),lr()/static_cast<double>(lr.max())};
        // Create double r= sin(first comp.)*12.5 * sin(second comp.)*12.5, for each pair of points.
        double r = sin(x[i][0]*12.5)*sin(x[i][1]*12.5);
        // If r>0 asign to output y=1, otherwise asign -1. (? operator works as if-else).
        // This is equivalent to: if (r>0){y[i][0]=1}; else{y[i][0]=-1};
        y[i][0] = (r>0)?1:-1;
    }
}

// Generate 1000 points of test data in a spiral pattern
void GetSpiralData(std::vector<MVector> &x, std::vector<MVector> &y)
{
    std::mt19937 lr;
    x = std::vector<MVector>(1000, MVector(2));
    y = std::vector<MVector>(1000, MVector(1));

    double twopi = 8.0*atan(1.0);
    for (int i=0; i<1000; i++)
    {
        x[i]={lr()/static_cast<double>(lr.max()),lr()/static_cast<double>(lr.max())};
        double xv=x[i][0]-0.5, yv=x[i][1]-0.5;
        double ang = atan2(yv,xv)+twopi;
        double rad = sqrt(xv*xv+yv*yv);

        double r=fmod(ang+rad*20, twopi);
        y[i][0] = (r<0.5*twopi)?1:-1;
    }
}

// Save the the training data in x and y to a new file, with the filename given by "filename"
// Boolean type -> Returns true if the file was saved succesfully
bool ExportTrainingData(const std::vector<MVector> &x, const std::vector<MVector> &y,
                        std::string filename)
{
    // Check that the training vectors are the same size
    assert(x.size()==y.size());

    // Open a file with the specified name.
    std::ofstream f(filename);

    // Return false, indicating failure, if file did not open
    if (!f) {return false;}

    // Loop over each training datum
    for (unsigned i=0; i<x.size(); i++)
    {
        // Check that the output for this point is a scalar
        assert(y[i].size() == 1);
        
        // Output components of x[i]
        for (int j=0; j<x[i].size(); j++)
        {
            f << x[i][j] << " ";
        }

        // Output only component of y[i]
        f << y[i][0] << " " << std::endl;
    }
    f.close();

    if (f) return true;
    else return false;
}

////////////////////////////////////////////////////////////////////////////////
// Neural network class

class Network
{
private:
    // Private member data
    std::vector<unsigned> nneurons;
    std::vector<MMatrix> weights;
    std::vector<MVector> biases, errors, activations, inputs;
    unsigned nLayers;

public:

    // Constructor: sets up vectors of MVectors and MMatrices for
    // weights, biases, weighted inputs, activations and errors
    // The parameter nneurons_ is a vector defining the number of neurons at each layer.
    // For example:
    //   Network({2,1}) has two input neurons, no hidden layers, one output neuron
    //
    //   Network({2,3,3,1}) has two input neurons, two hidden layers of three neurons
    //                      each, and one output neuron
    Network(std::vector<unsigned> nneurons_)
    // Pass a vector nneurons_ though contructor, with neuron specification.
    {
        nneurons = nneurons_;                    // Vector of neuron specification.
        nLayers = nneurons.size();               // Num of layers.
        
        weights = std::vector<MMatrix>(nLayers); // Vector (of matrices) of size nLayers, initialised at all zeros.
        
        // The following are all vectors (of vectors) of size nLayers:
        biases = std::vector<MVector>(nLayers);
        errors = std::vector<MVector>(nLayers);      // Derivative of cost function wrt input z_j^{[l]}.
        activations = std::vector<MVector>(nLayers); // Output after passing through activation function sigma=tanh z
        inputs = std::vector<MVector>(nLayers);      // inputs z_j^{[l]}
        
        // Create activations vector for input layer 0
        activations[0] = MVector(nneurons[0]);       // Vector of activations (outputs after sigma) of input layer.

        // Other vectors initialised for second and subsequent layers
        for (unsigned i=1; i<nLayers; i++)
        {
            // To ith entry of weights vector, assign zero matrix of nrow=n_i and ncol=n_{i-1}
            weights[i] = MMatrix(nneurons[i], nneurons[i-1]);
            // To ith entry of biases/inputs/erros/activations vector, assign vector of zeros size=n_i
            biases[i] = MVector(nneurons[i]);
            inputs[i] = MVector(nneurons[i]);
            errors[i] = MVector(nneurons[i]);
            activations[i] = MVector(nneurons[i]);
        }

        // The correspondence between these member variables and
        // the LaTeX notation used in the project description is:
        //
        //         C++           |      LaTeX      |                       Meaning:                           |
        // ---------------------------------------------------------------------------------------------------|
        // inputs[l-1][j-1]      |  z_j^{[l]}      |  Input of jth neuron at layer l                          |
        // activations[l-1][j-1] |  a_j^{[l]}      |  Output after passing input through act. function sigma. |
        // weights[l-1](j-1,k-1) |  W_{jk}^{[l]}   |  Weight matrix elements for jth neuron at layer l.       |
        // biases[l-1][j-1]      |  b_j^{[l]}      |  Biases vector entries for jth neuron at layer l.        |
        // errors[l-1][j-1]      |  \delta_j^{[l]} |  Derivative of cost function wrt input z_j^{[l]}.        |
        // nneurons[l-1]         |  n_l            |  Number of neurons at layer l                            |
        // nLayers               |  L              |  Number of layers                                        |
        
        // * Note that, since C++ vector indices run from 0 to N-1, all the indices in C++
        // code are one less than the indices used in the mathematics (which run from 1 to N)
    }

    // Return the number of input neurons n_1
    unsigned NInputNeurons() const
    {
        return nneurons[0];
    }

    // Return the number of output neurons n_L
    unsigned NOutputNeurons() const
    {
        return nneurons[nLayers-1];
    }
    
    // Evaluate the network for an input x and return the activations of the output layer
    MVector Evaluate(const MVector &x)
    {
        // Call FeedForward(x) to evaluate the network for an input vector x.
        // Feedforward should loop through all neurons and return activations after all layers.
        FeedForward(x);
        
        // Return the (final) activations of the output layer
        return activations[nLayers-1];
    }

    
    // Implement the training algorithm outlined in section 1.3.3
    // This should be implemented by calling the appropriate private member functions, below
    bool Train(const std::vector<MVector> x, const std::vector<MVector> y,
               double initsd, double learningRate, double costThreshold, int maxIterations)
    {
        // Check that there are the same number of training data inputs as outputs
        assert(x.size() == y.size());
       
        // Initialise the weights and biases with the standard deviation "initsd"
        InitialiseWeightsAndBiases(initsd);
        
        // Write iterations for convergence to datafile
        // std::ofstream DataFile;
        
        // DataFile.open("Num_Iter_initsd_point01.txt" /*, std::fstream::app */ );     // fstream::app -> Write on top of file.
        // DataFile << "Standard Deviation used: " << initsd << std::endl; // Keep track of SD
        
        // DataFile.width(15); DataFile << "Iterations";
        // DataFile.width(15); DataFile << "Total Cost" << std::endl;
        
        for (int iter=1; iter<=maxIterations; iter++)
        {
            // Step 3: Choose a random training data point i in {0, 1, 2, ..., N}
            int i = rnd()%x.size();

            // Run the feed-forward algorithm
            FeedForward(x[i]);

            // Run the back-propagation algorithm
            BackPropagateError(y[i]);
            
            // Update the weights and biases using stochastic gradient:
            UpdateWeightsAndBiases(learningRate);

            // Every so often (1000 iter), show an update on how the cost function has decreased:
            if ((!(iter%1000)) || iter==maxIterations)
            {
                // Calculate the total cost:
                double TotCost=TotalCost(x, y);
                
                // Display the iteration number and total cost to the screen
                std::cout << "Iteration: " << iter << std::endl;
                std::cout << "Total Cost: " << TotCost << std::endl;
                
                // if (iter==maxIterations) {DataFile.width(15); DataFile << "Fail" << std::endl;}
               
                // DataFile.width(15); DataFile << iter;
                // DataFile.width(15); DataFile << TotCost<< std::endl;
                
                // Return from this method with a value of true, if this cost is less than "costThreshold".
                if (TotCost < costThreshold) {
                    std:: cout << "Succesfully trained network in " << iter << " iterations." << std::endl;
                    // Write Number of iterations to file:
                //    DataFile.width(15); DataFile << iter << std::endl;
                    return true; }
            }
        } // Step 8: go back to step 3, until we have taken "maxIterations" steps

        // Step 9: return "false", indicating that the training did not succeed.
        return false;
    }

    
    // For a neural network with two inputs x=(x1, x2) and one output y,
    // loop over (x1, x2) for a grid of points in [0, 1]x[0, 1]
    // and save the value of the network output y evaluated at these points
    // to a file. Returns true if the file was saved successfully.
    bool ExportOutput(std::string filename)
    {
        // Check that the network has the right number of inputs and outputs
        assert(NInputNeurons()==2 && NOutputNeurons()==1);
    
        // Open a file with the specified name.
        std::ofstream f(filename);
    
        // Return false, indicating failure, if file did not open
        if (!f)
        {
            return false;
        }

        // generate a matrix of 250x250 output data points
        for (int i=0; i<=250; i++)
        {
            for (int j=0; j<=250; j++)
            {
                MVector out = Evaluate({i/250.0, j/250.0});
                f << out[0] << " ";
            }
            f << std::endl;
        }
        f.close();
    
        if (f) return true;
        else return false;
    }


    static bool Test();
    
private:
    // Return the activation function sigma:
    double Sigma(double z) {return tanh(z);}
    
    // Return the derivative of the activation function:
    double SigmaPrime(double z)
    {return 1.0-tanh(z)*tanh(z); }
    
    
    // Loop over all weights and biases in the network and set each
    // term to a random number normally distributed with mean 0 and standard deviation "initsd"
    void InitialiseWeightsAndBiases(double initsd)
    {
        // Make sure the standard deviation supplied is non-negative
        assert(initsd>=0);

        // Set up a normal distribution with mean zero, standard deviation "initsd"
        // Calling "dist(rnd)" returns a random number drawn from this distribution.
        std::normal_distribution<> dist(0, initsd);
        
        // Biases assigned at hidden and output layers. (Not for input layer)
        // Weights assigned after each neuron.
        for (int l=1; l< nLayers; l++) {                         // Loop from 1st non-input layer to output Layer.
            MVector this_bias(nneurons[l]);                      // Create zero MVector for a bias.
            MMatrix this_weight(nneurons[l], nneurons[l-1]);     // Creates a MMatrix of zeros.
            
            for (int j=0; j<nneurons[l]; j++)                    // Loop over all neurons per layer.
            {this_bias[j]+=dist(rnd);                            //
                
                for (int k=0;  k<nneurons[l-1]; k++)
                {this_weight(j,k)+=dist(rnd); }
            }
            biases[l]=this_bias;
            weights[l]=this_weight;
        }
    }
    

    // Evaluate the feed-forward algorithm, setting weighted inputs and activations
    // at each layer, given an input vector x
    void FeedForward(const MVector &x)
    {
        // Check that the input vector has the same number of elements as the input layer
        assert(x.size() == nneurons[0]);
        
        // Equation 1.8: Specify input; Sigma not evaluated here.
        inputs[0]=x;
        activations[0]=x;
        
        //Equation 1.7: Compute activations using act. function, weights and biases.
        for (int l=1; l<nLayers; l++){
            MVector wtimesa=weights[l]*activations[l-1];  // MMatrix times MVector returns MVector
            for (int j=0; j<nneurons[l]; j++) {
                inputs[l][j]=wtimesa[j]+biases[l][j];     // Specify inputs for later use in back-propagation algorithm.
                activations[l][j]=Sigma(inputs[l][j]);
            }
        }
    }

    // Evaluate the back-propagation algorithm, setting errors for each layer
    void BackPropagateError(const MVector &y)
    {
        // Check that the output vector y has the same number of elements as the output layer
        assert(y.size() == nneurons[nLayers - 1]);
        
        // Equation 1.22: Find error on output layer L (nLayers)
        for (int j=0; j<nneurons[nLayers-1]; j++) {
            errors[nLayers-1][j]=SigmaPrime(inputs[nLayers-1][j])*(activations[nLayers-1][j]-y[j]);
        }
        
        // Equation 1.24: Find error on output layer L (nLayers)
        for (int l=nLayers-2; l>=1; l--){
            MVector TT=TransposeTimes(weights[l+1],errors[l+1]);
            for (int j=0; j<nneurons[l]; j++)
            {errors[l][j]=SigmaPrime(inputs[l][j])*TT[j];}
        }
    }

    // Apply one iteration of the stochastic gradient iteration with learning rate eta.
    void UpdateWeightsAndBiases(double eta)
    {
        // Check that the learning rate is positive
        assert(eta>0);
        
        // Update the weights and biases according to the stochastic gradient iteration.
        
        // Biases -= eta delta_j^l
        for (int l=1; l<nLayers; l++)
        { biases[l]-=eta*errors[l]; }

        // Weights -= eta delta_j^l * a_k[l-1]
        for (int l=1; l<nLayers; l++)
        {
            MMatrix OP=OuterProduct(errors[l], activations[l-1]);
            weights[l]-=eta*OP; }
        }
    
    // Return the cost function of the network with respect to a single the desired output y
    // Note: call FeedForward(x) first to evaluate the network output for an input x,
    //       then call this method Cost(y) with the corresponding desired output y
    double Cost(const MVector &y)
    {
        // Check that y has the same number of elements as the network has outputs
        assert(y.size() == nneurons[nLayers-1]);
        
        // TODO: Return the cost associated with this output
        
        // Create Diff vector storing the difference: y - activations[output layer]
        MVector Diff=y-activations[nLayers-1];
        
        // Compute the L2 Norm for Diff Vector:
        double sum=0;
        for (int i=0; i<Diff.size(); i++) {sum+=Diff[i]*Diff[i];}
        
        return 1./2.*sum;
    }

    // Return the total cost C for a set of training data x and desired outputs y
    double TotalCost(const std::vector<MVector> x, const std::vector<MVector> y)
    {
        // Check that there are the same number of inputs as outputs
        assert(x.size() == y.size());
        // TODO: Implement the cost function, equation (1.9), using
        //       the FeedForward(x) and Cost(y) methods
        
        double sum=0.0;
        
        for (int i=0; i<x.size(); i++) {
            FeedForward(x[i]);
            double this_cost=Cost(y[i]);
            sum+=this_cost;
        }
        double N=x.size();
        return 1.0/N*sum;
    }

};

bool Network::Test()
{
    // This function is a static member function of the Network class:
    // it acts like a normal stand-alone function, but has access to private
    // members of the Network class. This is useful for testing, since we can
    // examine and change internal class data.
    //
    // This function should return true if all tests pass, or false otherwise.
    //
    // The following tests will be carried out:
    // 1. Testing FeedForward on a simple neural network.
    // 2. Testing biases/weights randomisation confirming data within 6 sigma.
    // 3. Testing BackPropagation on a simple neural network.

    // 1. A example test of FeedForward
    { // Make a simple network with two weights and one bias
        Network n({2, 1});

        // Set the values of these by hand
        n.biases[1][0] = 0.5; n.weights[1](0,0) = -0.3; n.weights[1](0,1) = 0.2;
        
        // Call function to be tested with x = (0.3, 0.4)
        n.FeedForward({0.3, 0.4});

        // Display the output value calculated
        std::cout << n.activations[1][0] << std::endl;

        // Correct value is = tanh(0.5 + (-0.3*0.3 + 0.2*0.4))
        //                    = 0.454216432682259...
        // Fail if error in answer is greater than 10^-10:
        if (std::abs(n.activations[1][0] - 0.454216432682259) > 1e-10){return false;}
        else {std::cout << "Successfuly tested FeedForward Algorithm" << std::endl;}
        
        MVector y={1.0};
        n.BackPropagateError(y);
        std::cout << "Error in output node:" << n.errors[1] << std::endl;
        
    }
    
    // 2. Test sigma and sigma prime work as expected.
    {   Network n({2, 1});
        if (std::abs(n.Sigma(1.0) - std::tanh(1.0)) >1e-6 ) {return false;}
        if (std::abs(n.SigmaPrime(1.0) - sech(1.0)*sech(1.0)) >1e-6 ) {return false;}
        std::cout << "Succesfully tested Sigma and Simga Prime" << std::endl;
    }
    
    // 3. Test weights / bias randomisation: Confirm randomisation lies within +-3 sigma.
    {
        // Create network with 3 layers and: 6 weights: {4,2}, 3 biases : {2,1}.
        Network n({2, 2, 1});

        // Set the values of these by hand
        double std_dev=0.1;
        n.InitialiseWeightsAndBiases(std_dev);
        
        bool print=false; // Set this to true, to print layer-by layer weights and biases.
        if (print==true)
        {   std::cout << "Biases Layer 2: " << std::endl;
            std::cout << n.biases[1] << std::endl; std::cout << "" << std::endl;
            std::cout << "Biases Layer 3: " << std::endl;
            std::cout << n.biases[2] << std::endl; std::cout << "" << std::endl;
            std::cout << "Weights Layer 2: " << std::endl;
            std::cout << n.weights[1] << std::endl;
            std::cout << "Weights Layer 3: " << std::endl;
            std::cout << n.weights[2] << std::endl; }
        
        // Weights and Biases should lie (with 99.7% certainty) within +- 3 sigma of the mean=0:
        // I.e. Weights and Biases must lie in (-0.3 , +0.3). Otherwise fail this test.
        for (int l=1; l<n.nLayers; l++) {
            for (int j=0; j<n.nneurons[l]; j++) {
                if (n.biases[l][j] < -3.0*std_dev || n.biases[l][j] > 3.0*std_dev ) // Confirm biases are OK
                {return false;}
                for (int k=0; k<n.nneurons[l-1]; k++) {
                    if (n.weights[l](j,k) < -3.0*std_dev || n.weights[l](j,k) > 3.0*std_dev )
                    {return false;}                   }  }   }
      std::cout << "Successfuly tested weight / bias randomisation" << std::endl;
    }
    
    // 4. Test BackForward Propagation / Update weights and biases on simple network.
    {   Network n({2, 2, 1});
        // Set weights / biases manually:
        n.biases[1][0] = 0.5; n.biases[1][1] = 0.5; n.biases[2][0] = 0.5;
        n.weights[1](0,0) = 0.3; n.weights[1](0,1) = 0.3; n.weights[1](1,0) = 0.3; n.weights[1](1,1) = 0.3;
        n.weights[2](0,0) = 0.3; n.weights[2](0,1) = 0.3;
            
        // Call feed-forward with x = (1, 1)
        n.FeedForward({1.0, 1.0});
        
        // Test Inputs:
        if (n.inputs[1][0] != 1.1) {return false;}
        double exp_final_input = 2*tanh(1.1)*0.3+0.5;
        if (abs(n.inputs[2][0]-exp_final_input)>1e-8) {return false;}
        
        // Further Test on FeedForward:
        double exp_output=tanh(exp_final_input);
        if (abs(exp_output-n.activations[2][0])>1e-8) {return false;}
        
        n.BackPropagateError({1.0});
        // Test BackPropagate:
        double exp_final_err=sech(exp_final_input)*sech(exp_final_input)*(exp_output-1.0);
        if (abs(n.errors[2][0]-exp_final_err)> 1e-8) {return false;}
        
        double exp_hidden_err=0.3*exp_final_err*sech(1.1)*sech(1.1);
        if (abs(n.errors[1][0]-exp_hidden_err)> 1e-8) {return false;}
        
        std::cout << "Sucesfully tested BackPropagation" << std::endl;
        
        n.UpdateWeightsAndBiases(1.0);
        // Test UpdateWeights and Biases:
        if (abs(n.biases[2][0]-(0.5-exp_final_err)) > 1e-8) {return false;}
        if (abs(n.biases[1][0]-(0.5-exp_hidden_err)) > 1e-8) {return false;}
        if (abs(n.weights[2](0,0)-(0.3-exp_final_err*tanh(1.1))) > 1e-8) {return false;}
        if (abs(n.weights[1](0,0)-(0.3-exp_hidden_err)) > 1e-8) {return false;}
  
        std::cout << "Sucesfully tested UpdateWeightsAndBiases" << std::endl;
    }
    
    
    // 5. Test Cost and Total Cost.
    {   Network n({2,2});
        n.weights[1](0,0) = 1.0; n.weights[1](0,1) = 1.0;
        n.weights[1](1,0) = 1.0; n.weights[1](1,1) = 1.0;
        
        // Single Datum:
        n.FeedForward({0.5, 0.5}); // Output {tanh(0.5+0.5), tanh(0.5+0.5)}
        if (abs(n.Cost({1.0, 1.0})-pow(tanh(1)-1, 2))> 1e-8) {return false;}
        std::cout << "Sucesfully tested cost" << std::endl;
        
        MVector tp1={0.5, 0.5}, tp2={0.75, 0.75}, out1={1.0, 1.0}, out2={2.0, 2.0};
        std::vector<MVector> x={tp1, tp2}, y={out1, out2};
        double Exp_Tot_Cost = 0.5*(pow(tanh(1)-1, 2)+pow(tanh(1.5)-2, 2));
        if (abs(Exp_Tot_Cost-n.TotalCost(x, y))> 1e-8) {return false;}
        std::cout << "Sucesfully tested total cost" << std::endl;
    }
    
    // TODO: for each part of the Network class that you implement,
    //       write some more tests here to run that code and verify that
    //       its output is as you expect.
    //       I recommend putting each test in an empty scope { ... }, as
    //       in the example given above.

    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Main function and example use of the Network class

// Create, train and use a neural network to classify the data in
// figures 1.1 and 1.2 of the project description.
//
// You should make your own copies of this function and change the network parameters
// to solve the other problems outlined in the project description.
double ClassifyTestData()
{
    // Create a network with two input neurons, two hidden layers of three neurons, and one output neuron
    Network n({2, 20, 20, 20, 1});
    
    // Get some data to train the network
    std::vector<MVector> x, y;
    
    // GetTestData(x, y);
    //GetCheckerboardData(x, y);
     GetSpiralData(x, y);
    
    // Train network on training inputs x and outputs y
    // Numerical parameters are:
    //  initial weight and bias standard deviation = 0.1
    //  learning rate = 0.1
    //  cost threshold = 1e-4
    //  maximum number of iterations = 10000
    double iter=0;
    bool trainingSucceeded = n.Train(x, y, 0.1, 0.01, 0.1, 1000000);
    
    // If training failed, report this
    if (!trainingSucceeded)
    {std::cout << "Failed to converge to desired tolerance." << std::endl;}
    
    
    // Generate some output files for plotting
    ExportTrainingData(x, y, "test_points_spiral.txt");
    n.ExportOutput("test_contour_spiral.txt");
    
    return iter;
}

int main()
{
    // Call the test function
    bool testsPassed = Network::Test();

    // If tests did not pass, something is wrong; end program now
    if (!testsPassed)
    {
        std::cout << "A test failed." << std::endl;
        return 1;
    }


    ClassifyTestData();
    
    return 0;
}

