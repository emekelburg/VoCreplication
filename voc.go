// create go-mod file with go mod init <module-name>
// create sharable library with:  go build -buildmode=c-shared -o libgeneticalgorithm.so geneticalgorithm.go
// create standalone executable for restful interaction:  go build -o geneticalgorithmapi geneticalgorithm.go

// required  gonum:  go get gonum.org/v1/gonum/mat
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"runtime"
	"sync"

	//"math/rand"
	"net/http"
	"time"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distmv"
)

// RequestData structure to hold incoming JSON data
type RequestData struct {
	Y          []float64   `json:"Y"`
	X          [][]float64 `json:"X"`
	LambdaList []float64   `json:"lambdaList"`
}

type StandardizeRequest struct {
	K      [][]float64 `json:"K"`
	Window *int        `json:"window,omitempty"`
}
type StandardizeResponse struct {
	Kout   [][]float64 `json:"Kout"`
	Kscale [][]float64 `json:"Kscale"`
}

// BenchmarkRequest struct now has pointers for optional boolean fields
type BenchmarkRequest struct {
	X      [][]float64 `json:"X"`
	Y      [][]float64 `json:"Y"`
	TrnWin int         `json:"trnWin"`
	Stdize *bool       `json:"stdize,omitempty"` // Pointer to make it optional
	Demean *bool       `json:"demean,omitempty"` // Pointer to make it optional
}

// Define a struct for your response
type BenchmarkResponse struct {
	YPrd         [][]float64 `json:"YPrd"`
	YPrdRescaled [][]float64 `json:"YPrdRescaled"`
	BetaNorms    [][]float64 `json:"BetaNorms"`
	ErrorMessage string      `json:"errorMessage,omitempty"`
}

// BenchmarkRequest struct now has pointers for optional boolean fields
type ForecastRequest struct {
	X      [][]float64 `json:"X"`
	Y      [][]float64 `json:"Y"`
	TrnWin int         `json:"trnWin"`
	Stdize *bool       `json:"stdize,omitempty"` // Pointer to make it optional
	Demean *bool       `json:"demean,omitempty"` // Pointer to make it optional
}

// Define a struct for your response
type ForecastResponse struct {
	YPrd         [][][]float64 `json:"YPrd"`
	YPrdRescaled [][][]float64 `json:"YPrdRescaled"`
	BetaNorms    [][][]float64 `json:"BetaNorms"`
	ErrorMessage string        `json:"errorMessage,omitempty"`
}

func main() {
	http.HandleFunc("/kellyridge", kellyRidgeHandler)
	http.HandleFunc("/regularridge", regularRidgeHandler)
	http.HandleFunc("/standardize", standardizeHandler)
	http.HandleFunc("/benchmarksim", benchmarkSimHandler)
	http.HandleFunc("/rffridgesim", RFFRidgeSimHandler)
	log.Fatal(http.ListenAndServe(":9008", nil))
}

// Webservice Request Handlers

// ConvertSlicesToMat converts a slice of slices of float64 to a *mat.Dense matrix.
func slicesToMat(slices [][]float64) *mat.Dense {
	if len(slices) == 0 || len(slices[0]) == 0 {
		// Return an empty matrix if the input slice is empty or the first row is empty.
		return mat.NewDense(0, 0, nil)
	}

	numRows := len(slices)
	numCols := len(slices[0])

	// Flatten the slice of slices to a single slice since mat.NewDense expects data in this format.
	data := make([]float64, 0, numRows*numCols)
	for _, row := range slices {
		if len(row) != numCols {
			// Optionally, handle variable-length rows here, though typically,
			// all rows should have the same number of columns for a well-formed matrix.
			panic("all rows must have the same number of columns")
		}
		data = append(data, row...)
	}

	// Create a new *mat.Dense matrix with the provided data.
	return mat.NewDense(numRows, numCols, data)
}

func standardizeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Only POST method is accepted", http.StatusMethodNotAllowed)
		return
	}

	var req StandardizeRequest
	err := json.NewDecoder(r.Body).Decode(&req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	//Kout, Kscale := volstdbwd(req.K, req.Window) // Call your function

	// Convert requestData into mat.Dense matrix
	K_ := slicesToMat(req.K)

	Kout, Kscale := volstdbwd(K_, req.Window)

	resp := StandardizeResponse{
		Kout:   matToSlices(Kout),
		Kscale: matToSlices(Kscale),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func regularRidgeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Only POST method is accepted", http.StatusMethodNotAllowed)
		return
	}

	var requestData RequestData
	err := json.NewDecoder(r.Body).Decode(&requestData)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Convert requestData.X and requestData.Y into mat.Dense matrices
	/*	rows, cols := len(requestData.X), len(requestData.X[0])
		flatX := make([]float64, 0, rows*cols)
		for _, row := range requestData.X {
			flatX = append(flatX, row...)
		}
		X := mat.NewDense(rows, cols, flatX)
		Y := mat.NewDense(len(requestData.Y), 1, requestData.Y) */
	X := slicesToMat(requestData.X)
	Y := mat.NewDense(len(requestData.Y), 1, requestData.Y)

	// Perform ridge regression
	B, err := calculateBetasUsingStandardRidge(Y, X, requestData.LambdaList)
	//B, err := calculateBetasUsingKellyMethod(Y, X, requestData.LambdaList)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Convert the coefficients matrix to a slice of slices for JSON encoding
	BData := make([][]float64, B.RawMatrix().Cols)
	for i := range BData {
		BData[i] = make([]float64, B.RawMatrix().Rows)
		for j := range BData[i] {
			BData[i][j] = B.At(j, i)
		}
	}

	// Send as part of a JSON response
	response := map[string]interface{}{"coefficients": BData}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func kellyRidgeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Only POST method is accepted", http.StatusMethodNotAllowed)
		return
	}

	var requestData RequestData
	err := json.NewDecoder(r.Body).Decode(&requestData)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Convert requestData.X and requestData.Y into mat.Dense matrices
	rows, cols := len(requestData.X), len(requestData.X[0])
	flatX := make([]float64, 0, rows*cols)
	for _, row := range requestData.X {
		flatX = append(flatX, row...)
	}
	X := mat.NewDense(rows, cols, flatX)
	Y := mat.NewDense(len(requestData.Y), 1, requestData.Y)

	// Perform ridge regression
	//B, err := calculateBetasUsingStandardRidge(Y, X, requestData.LambdaList)
	B, err := calculateBetasUsingKellyMethod(Y, X, requestData.LambdaList)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Convert the coefficients matrix to a slice of slices for JSON encoding
	BData := make([][]float64, B.RawMatrix().Cols)
	for i := range BData {
		BData[i] = make([]float64, B.RawMatrix().Rows)
		for j := range BData[i] {
			BData[i][j] = B.At(j, i)
		}
	}

	// Send as part of a JSON response
	response := map[string]interface{}{"coefficients": BData}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// benchmarkSimHandler wraps your benchmarkSim function for HTTP access
func benchmarkSimHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST method is accepted", http.StatusMethodNotAllowed)
		return
	}

	var req BenchmarkRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Set default values for Stdize and Demean if they are nil
	stdize := true // Default value for Stdize
	if req.Stdize != nil {
		stdize = *req.Stdize
	}

	demean := false // Default value for Demean
	if req.Demean != nil {
		demean = *req.Demean
	}

	// Convert the request data to *mat.Dense types
	Xmat := mat.NewDense(len(req.X), len(req.X[0]), nil)
	for i, row := range req.X {
		for j, val := range row {
			Xmat.Set(i, j, val)
		}
	}

	Ymat := mat.NewDense(len(req.Y), len(req.Y[0]), nil)
	for i, row := range req.Y {
		for j, val := range row {
			Ymat.Set(i, j, val)
		}
	}

	// Call the benchmarkSim function
	YPrd, YPrdRescaled, BetaNorms, err := benchmarkSim(Xmat, Ymat, req.TrnWin, stdize, demean)
	if err != nil {
		resp := BenchmarkResponse{ErrorMessage: err.Error()}
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(resp)
		return
	}

	// Prepare and send the response
	resp := BenchmarkResponse{
		YPrd:         matToSlices(YPrd),
		YPrdRescaled: matToSlices(YPrdRescaled),
		BetaNorms:    matToSlices(BetaNorms),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// benchmarkSimHandler wraps your benchmarkSim function for HTTP access
func RFFRidgeSimHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST method is accepted", http.StatusMethodNotAllowed)
		return
	}

	var req ForecastRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Set default values for Stdize and Demean if they are nil
	stdize := true // Default value for Stdize
	if req.Stdize != nil {
		stdize = *req.Stdize
	}

	demean := false // Default value for Demean
	if req.Demean != nil {
		demean = *req.Demean
	}

	// Convert the request data to *mat.Dense types
	Xmat := mat.NewDense(len(req.X), len(req.X[0]), nil)
	for i, row := range req.X {
		for j, val := range row {
			Xmat.Set(i, j, val)
		}
	}

	Ymat := mat.NewDense(len(req.Y), len(req.Y[0]), nil)
	for i, row := range req.Y {
		for j, val := range row {
			Ymat.Set(i, j, val)
		}
	}

	// Call the benchmarkSim function
	YPrd, YPrdRescaled, BetaNorms, err := RffRidgeSim(Xmat, Ymat, 1, 12000, 2, req.TrnWin, stdize, demean)
	/*	if err != nil {
			resp := ForecastResponse{ErrorMessage: err.Error()}
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(resp)
			return
		}

		// Prepare and send the response
		resp := ForecastResponse{
			YPrd:         matToSlices(YPrd),
			YPrdRescaled: matToSlices(YPrdRescaled),
			BetaNorms:    matToSlices(BetaNorms),
		}
	*/

	// Inside RFFRidgeSimHandler, after calling RffRidgeSim:
	if err != nil {
		resp := ForecastResponse{ErrorMessage: err.Error()}
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(resp)
		return
	}

	// Prepare and send the response
	resp := ForecastResponse{
		YPrd:         matToSlices3D(YPrd),
		YPrdRescaled: matToSlices3D(YPrdRescaled),
		BetaNorms:    matToSlices3D(BetaNorms),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// matToSlices converts a *mat.Dense matrix to a slice of slices for JSON encoding
func matToSlices(m *mat.Dense) [][]float64 {
	r, c := m.Dims()
	out := make([][]float64, r)
	for i := range out {
		out[i] = make([]float64, c)
		for j := range out[i] {
			out[i][j] = m.At(i, j)
		}
	}
	return out
}

// matToSlices3D converts a slice of *mat.Dense matrices to a three-dimensional slice of float64.
func matToSlices3D(mats []*mat.Dense) [][][]float64 {
	var result [][][]float64
	for _, mat := range mats {
		result = append(result, matToSlices(mat))
	}
	return result
}

// Fuctions for Standardization

// volstdbwd standardizes a matrix by volatility using an expanding or rolling window.
// If no window is provided, it uses an expanding window with a minimum of 36 observations,
// or all available observations if fewer than 36 are present.
func volstdbwd(K *mat.Dense, window *int) (*mat.Dense, *mat.Dense) {
	rows, cols := K.Dims()

	Kout := mat.NewDense(rows, cols, nil)
	Kscale := mat.NewDense(rows, cols, nil)

	var initialWindowSize int
	if window == nil {
		initialWindowSize = min(rows, 36) // Use up to 36 for expanding window if no window size is provided
	} else {
		initialWindowSize = *window
	}

	for p := 0; p < cols; p++ {
		data := make([]float64, rows)
		for t := 0; t < rows; t++ {
			data[t] = K.At(t, p)
		}

		var initialStdDev float64
		if len(data) == 0 {
			initialStdDev = 1.0 // Default to prevent division by zero
		} else {
			initialStdDev = stat.StdDev(data[:initialWindowSize], nil)
		}

		for t := 0; t < rows; t++ {
			var stdDev float64
			if t < initialWindowSize {
				stdDev = initialStdDev
			} else if window == nil {
				stdDev = stat.StdDev(data[:t+1], nil)
			} else {
				start := max(0, t-*window+1)
				stdDev = stat.StdDev(data[start:t+1], nil)
			}

			if stdDev != 0 {
				Kout.Set(t, p, K.At(t, p)/stdDev)
				Kscale.Set(t, p, stdDev)
			} else {
				Kout.Set(t, p, 0)   // Handle division by zero if stdDev is zero
				Kscale.Set(t, p, 1) // Default scale factor to 1 if stdDev is zero
			}
		}
	}

	return Kout, Kscale
}

// the same function using a simple float64 array as inputs and outputs
//
//	duplicative to the previous function, and only here for potential future use in other scenarios
func volstdbwd_float(K [][]float64, window *int) ([][]float64, [][]float64) {
	T := len(K)
	P := len(K[0]) // Assuming all columns have the same length

	Kout := make([][]float64, T)
	Kscale := make([][]float64, T)
	for i := range Kout {
		Kout[i] = make([]float64, P)
		Kscale[i] = make([]float64, P)
	}

	// Calculate the initial standard deviation for the first window or up to 36 observations
	var initialWindowSize int
	if window == nil {
		initialWindowSize = min(T, 36) // Use up to 36 for expanding window if no window size is provided
	} else {
		initialWindowSize = *window
	}

	initialStdDev := make([]float64, P)
	for p := 0; p < P; p++ {
		data := make([]float64, initialWindowSize)
		for t := 0; t < initialWindowSize; t++ {
			data[t] = K[t][p]
		}
		initialStdDev[p] = stat.StdDev(data, nil)
	}

	for p := 0; p < P; p++ {
		for t := 0; t < T; t++ {
			var stdDev float64
			if t < initialWindowSize {
				// Use the initial standard deviation for the first window size or up to 36 observations
				stdDev = initialStdDev[p]
			} else if window == nil {
				// Expanding window: Calculate standard deviation using all available observations up to t
				data := make([]float64, t+1)
				for i := 0; i <= t; i++ {
					data[i] = K[i][p]
				}
				stdDev = stat.StdDev(data, nil)
			} else {
				// Fixed rolling window: Calculate standard deviation for the current window
				start := max(0, t-*window+1)
				data := make([]float64, t-start+1)
				for i := start; i <= t; i++ {
					data[i-start] = K[i][p]
				}
				stdDev = stat.StdDev(data, nil)
			}

			Kscale[t][p] = stdDev
			if stdDev != 0 {
				Kout[t][p] = K[t][p] / stdDev
			} else {
				Kout[t][p] = 0 // Handle division by zero if stdDev is zero
			}
		}
	}

	return Kout, Kscale
}

// Fuctions for Coefficient Calculation

// calculateBetasUsingStandardRidge performs ridge regression using Singular Value Decomposition.
// Y is a matrix of shape (T, 1), X is a matrix of shape (T, P), and lambdaList is a slice of lambda values.
// It returns a matrix B of shape (P, L), where L is the length of lambdaList.
func calculateBetasUsingStandardRidge(Y, X *mat.Dense, lambdaList []float64) (*mat.Dense, error) {

	var svd mat.SVD
	// Factorize the matrix X using SVD with thin SVD option.
	ok := svd.Factorize(X, mat.SVDThin)
	if !ok {
		// Return an error if SVD factorization fails.
		return nil, fmt.Errorf("SVD factorization failed")
	}

	// U matrix from the SVD, where U is the matrix of left singular vectors.
	var U mat.Dense
	svd.UTo(&U) // Extract U matrix to a new Dense matrix U.
	//fmt.Println("U:", U)

	// VT (V transpose) matrix from the SVD, where V is the matrix of right singular vectors.
	var VT mat.Dense
	svd.VTo(&VT) // Extract V transpose matrix to a new Dense matrix VT.
	//fmt.Println("VT:", VT)

	// Singular values from the SVD.
	S := svd.Values(nil) // Retrieve singular values.
	//fmt.Println("S:", S)

	// P is the number of columns in X, L is the length of lambdaList.
	P, L := X.RawMatrix().Cols, len(lambdaList)
	// Initialize matrix B to store the regression coefficients.
	B := mat.NewDense(P, L, nil)

	// Loop through each lambda value in the lambda list.
	for index, lambda := range lambdaList {
		// Initialize a slice to store modified singular values for ridge regression.
		Dplus := make([]float64, len(S))
		for i, s := range S {
			// Modify singular values according to ridge regression formula.
			Dplus[i] = s / (s*s + lambda)
		}
		// Create a diagonal matrix from the modified singular values.
		diagS := mat.NewDiagDense(len(S), Dplus)

		// Temporary matrix to hold intermediate results.
		temp1 := &mat.Dense{}
		// Calculate the product of VT^T, diagS.
		//temp1.Product(VT.T(), diagS)
		temp1.Product(&VT, diagS)
		temp2 := &mat.Dense{}
		// Calculate the product of the above result with U^T and Y to get regression coefficients for current lambda.
		temp2.Product(temp1, U.T(), Y)

		// Set the calculated coefficients for the current lambda in the B matrix.
		for i := 0; i < P; i++ {
			B.Set(i, index, temp2.At(i, 0))
		}
	}

	// Return the matrix B containing regression coefficients for each lambda value.
	return B, nil
}

// calculateBetasUsingKellyMethod calculates regression coefficients using the Kelly method.
func calculateBetasUsingKellyMethod(Y, X *mat.Dense, lambdaList []float64) (*mat.Dense, error) {
	T_, P_ := X.Dims() // T = number of observations, P = number of parameters

	// Ensure P > T, otherwise use regular Ridge regression
	if P_ <= T_ {
		return nil, fmt.Errorf("P must be greater than T for the Kelly method")
	}

	/* Compute the covariance matrix of the observations (T x T) scaled by the number of observations.
	   The result is a T×T matrix.
	   This scaling factor (T_) normalizes the covariance matrix, making it essentially the
	   average covariance across all observations
	    WHY:  For high-dimensional data (P>T), where the number of predictors exceeds the number of observations,
	    it's more computationally efficient and numerically stable to work with a smaller T×T matrix rather
	    than a larger P×P matrix. */
	aMatrix := mat.NewDense(T_, T_, nil)
	aMatrix.Product(X, X.T())
	aMatrix.Scale(1/float64(T_), aMatrix)

	/* Perform SVD on the covariance matrix
		   This decomposes this covariance matrix into its eigenvalues (s) and eigenvectors (U and V).
	       The eigenvalues represent the variance explained by each principal component,
	       while the eigenvectors represent the directions of maximum variance in the data space.
	       This decomposition is crucial for understanding the structure of the data and for further
	       calculations in the function, particularly for regularization and dimensionality reduction purposes. */
	var svd mat.SVD
	ok := svd.Factorize(aMatrix, mat.SVDThin)
	if !ok {
		return nil, fmt.Errorf("SVD factorization failed")
	}
	// Extract the U matrix
	var U mat.Dense
	svd.UTo(&U)

	// and V^T matrix from the SVD - not needed for Kelly method
	//var VT mat.Dense
	//svd.VTo(&VT)

	// extract singular values (eigenvalues)
	S := svd.Values(nil)

	/* Make eigenvalues (amount of variance explained) into a scaled matrix with same dimensions as a_matrix
	   The below commands explicitly compute the inverse square root only for non-zero eigenvalues,
	   effectively bypassing the division by zero issue for zero eigenvalues by setting their inverse square roots to zero.
	*/
	invSqrtEigvals := make([]float64, len(S))
	for i, val := range S {
		if val > 0 {
			invSqrtEigvals[i] = 1 / math.Sqrt(val*float64(T_))
		}
	}
	// create square matrix of the results to ensure right dimensionality
	// Determine the dimension for the square matrix
	dim := len(invSqrtEigvals)
	// Create a full dense matrix of the appropriate size filled with zeros
	scaleEigval := mat.NewDense(dim, dim, nil)

	// Populate the diagonal with the scaled eigenvalues (inverse square roots)
	for i := 0; i < dim; i++ {
		scaleEigval.Set(i, i, invSqrtEigvals[i])
	}

	// Calculate W = X'U * scaleEigval
	/* W is constructed by multiplying X' (transpose of X, making it PxT), U (from the SVD, dimension depends on a_matrix),
	   and a scaling matrix derived from D_a (scale_eigval, which applies an inverse square root scaling to the eigenvalues).
	  This operation transforms the predictors into a space where the regularization will be applied. */
	W := mat.NewDense(P_, T_, nil) // P x T
	W.Product(X.T(), &U, scaleEigval)

	// Calculate signal * return = X'Y / T
	/* vector of dimension Px1 representing the correlation (or "signal") between each predictor and the response variable,
	   scaled by the number of observations */
	signalTimesReturn := mat.NewDense(P_, 1, nil)
	signalTimesReturn.Product(X.T(), Y)
	signalTimesReturn.Scale(1/float64(T_), signalTimesReturn)

	// Calculate signal * return * v = W' * signal * return
	// adjust the signal vector into the transformed predictor space
	signalTimesReturnTimesV := mat.NewDense(T_, 1, nil)
	signalTimesReturnTimesV.Product(W.T(), signalTimesReturn)

	// Initialize B to store the regression coefficients
	B := mat.NewDense(P_, len(lambdaList), nil)

	// iterate through all lambdas
	for index, lambda := range lambdaList {

		/* Adjust the transformed predictors (W) by the regularization term.
		This operation scales each transformed predictor component based on
		its eigenvalue and the regularization parameter */
		regularizedPredictors := mat.NewDense(P_, T_, nil) // P x T
		for i, val := range S {
			if val > 0 {
				invSqrtEigvals[i] = 1 / (val + lambda)
			} else {
				invSqrtEigvals[i] = 0
			}
		}

		regDiag := mat.NewDiagDense(len(S), invSqrtEigvals)
		regularizedPredictors.Product(W, regDiag)

		coefficients := mat.NewDense(P_, 1, nil)
		coefficients.Product(regularizedPredictors, signalTimesReturnTimesV)

		// Set coefficients into B
		for i := 0; i < P_; i++ {
			B.Set(i, index, coefficients.At(i, 0))
		}
	}

	return B, nil
}

// Simulation Functions

// generateLambdaList generates a list of lambda values for ridge regression.
// startPow and endPow define the range of powers of 10 for the lambda values,
// and step defines the step size between consecutive powers.
func generateLambdaList(startPow, endPow, step float64) []float64 {
	var lamlist []float64
	for pow := startPow; pow <= endPow; pow += step {
		lamlist = append(lamlist, math.Pow(10, pow))
	}
	return lamlist
}

// demeanVector demeans a vector (or a single column of a matrix) in place and returns its mean.
// It takes a vector as input, which could be a column of a matrix or any 1D slice.
func demeanVector(v []float64) float64 {
	sum := 0.0
	for _, value := range v {
		sum += value
	}
	mean := sum / float64(len(v))
	for i := range v {
		v[i] -= mean
	}
	return mean
}

// Function to add one lag of the Y-variable to the X-array
func addLaggedYToX(X, Y *mat.Dense) *mat.Dense {
	rowsX, colsX := X.Dims()
	rowsY, _ := Y.Dims()

	// Check if the number of rows in X and Y are equal
	if rowsX != rowsY {
		fmt.Println("Error: The number of rows in X and Y must be the same.")
		return nil
	}

	// Create a new slice to hold the lagged Y values with one position shifted
	laggedY := make([]float64, rowsY)
	// Assuming the first value of lagged Y to be 0 (or any appropriate placeholder)
	laggedY[0] = 0 // math.NaN()  --this is what matlab does (sets NaN value), but causes error without explicit handling in Go
	for i := 1; i < rowsY; i++ {
		laggedY[i] = Y.At(i-1, 0)
	}

	// Create a new matrix that is one column wider than X to accommodate the lagged Y
	XLaggedY := mat.NewDense(rowsX, colsX+1, nil)

	// Copy the original X matrix into the new matrix
	for i := 0; i < rowsX; i++ {
		for j := 0; j < colsX; j++ {
			XLaggedY.Set(i, j, X.At(i, j))
		}
	}

	// Add the lagged Y column to the new matrix
	for i := 0; i < rowsY; i++ {
		XLaggedY.Set(i, colsX, laggedY[i])
	}

	return XLaggedY
}

// computeStdDev returns the standard deviation of each column in a *mat.Dense matrix.
func computeStdDev(m *mat.Dense) []float64 {
	_, cols := m.Dims()
	stdDevs := make([]float64, cols)
	for j := 0; j < cols; j++ {
		col := mat.Col(nil, j, m)
		stdDevs[j] = stat.StdDev(col, nil)
		if stdDevs[j] == 0 {
			stdDevs[j] = 1 // Avoid division by zero
		}
	}
	return stdDevs
}

// scaleByStdDev scales each column of the matrix by the provided standard deviations.
func scaleByStdDev(m *mat.Dense, stdDevs []float64) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.Set(i, j, m.At(i, j)/stdDevs[j])
		}
	}
}

// scaleVecByStdDev scales a vector by the provided standard deviations.
func scaleVecByStdDev(v *mat.VecDense, stdDevs []float64) {
	len := v.Len()
	for i := 0; i < len; i++ {
		if stdDevs[i] == 0 {
			stdDevs[i] = 1 // Avoid division by zero
		}
		v.SetVec(i, v.AtVec(i)/stdDevs[i])
	}
}

// preprocessData standardizes X and Y, adds lagged Y to X, and drops initial observations.
func preprocessData(X, Y *mat.Dense, trnwin int, stdize bool) (*mat.Dense, *mat.Dense, *mat.Dense, error) {

	var Xstd, Ystd, YScaleStd *mat.Dense

	XLaggedY := addLaggedYToX(X, Y)

	if stdize {
		// volstdbwd standardizes X and Y and returns scaling factors
		Xstd, _ = volstdbwd(XLaggedY, nil)
		Ystd, YScaleStd = volstdbwd(Y, &trnwin)

		// Drop the first 36 observations from Xstd and Ystd
		rowsX, colsX := Xstd.Dims()
		rowsY, colsY := Ystd.Dims()

		// Ensure there are more than 36 observations
		if rowsX > 36 && rowsY > 36 {
			Xstd = Xstd.Slice(36, rowsX, 0, colsX).(*mat.Dense)
			Ystd = Ystd.Slice(36, rowsY, 0, 1).(*mat.Dense)
			YScaleStd = YScaleStd.Slice(36, rowsY, 0, colsY).(*mat.Dense) // Adjust YScaleStd similarly if applicable
		} else {
			fmt.Println("Warning: Less than 36 observations provided after standardization.")
			// Handle this scenario appropriately; perhaps by continuing with available data
			// or returning an error indicating insufficient data.
			return nil, nil, nil, nil
		}
	} else {
		Xstd = XLaggedY
		Ystd = Y
		// Create a dummy YScaleStd with ones if not standardizing
		_, cols := Y.Dims()
		YScaleStd = mat.NewDense(1, cols, nil)
		for i := 0; i < cols; i++ {
			YScaleStd.Set(0, i, 1.0)
		}
	}

	return Xstd, Ystd, YScaleStd, nil
}

/*
// performRollingWindowRidgeRegression encapsulates the rolling window Ridge regression logic.
func performRollingWindowRidgeRegression(Xstd, Ystd, YScaleStd *mat.Dense, trnwin int, demean bool, lamlist []float64, useKellyMethod bool) (*mat.Dense, *mat.Dense, *mat.Dense, error) {
	T, _ := Ystd.Dims()
	nLambda := len(lamlist)

	// _, cols := Xstd.Dims()
	YPrd := mat.NewDense(T, nLambda, nil)         // Unscaled predictions
	YPrdRescaled := mat.NewDense(T, nLambda, nil) // Rescaled predictions
	BNorm := mat.NewDense(T, nLambda, nil)        // Norms of the coefficients

	_, cols := Xstd.Dims()

	for t := trnwin; t < T; t++ {

		start := max(0, t-trnwin) // Calculate the start index to keep the window size constant
		end := t                  // The end index is exclusive, so using t directly works as intended

		// Adjust Ztrn and Ytrn to contain only data within the rolling window
		Ztrn := Xstd.Slice(start, end, 0, cols).(*mat.Dense) // Slices rows [start, t) from Xstd
		Ytrn := Ystd.Slice(start, end, 0, 1).(*mat.Dense)    // Slices rows [start, t) from Ystd

		var meanY float64
		if demean {
			// Demean Ytrn and store the mean for adding back mean after prediction
			YtrnCol := mat.Col(nil, 0, Ytrn)
			meanY = demeanVector(YtrnCol)
			// Copy the demeaned data back to Ytrn
			for i := 0; i < len(YtrnCol); i++ {
				Ytrn.Set(i, 0, YtrnCol[i])
			}

			// Demean Ztrn
			_, cols := Ztrn.Dims()
			for c := 0; c < cols; c++ {
				ZtrnCol := mat.Col(nil, c, Ztrn)
				_ = demeanVector(ZtrnCol) // Demean in place; mean values are not directly used here
				for r := 0; r < len(ZtrnCol); r++ {
					Ztrn.Set(r, c, ZtrnCol[r])
				}
			}
		}

		// Compute standard deviation for each column in Ztrn
		stdDevs := computeStdDev(Ztrn)
		// Scale Ztrn by its column-wise standard deviations
		scaleByStdDev(Ztrn, stdDevs)

		// Calculate betas using the ridge regression function
		//betas, _ := calculateBetasUsingStandardRidge(Ytrn, Ztrn, lamlist)

		// Check if Kelly's method should be used
		rowsZtrn, colsZtrn := Ztrn.Dims()
		var betas *mat.Dense
		var err error
		if useKellyMethod && colsZtrn > rowsZtrn {
			//fmt.Printf("Using Kelly Method")
			betas, err = calculateBetasUsingKellyMethod(Ytrn, Ztrn, lamlist)
		} else {
			//fmt.Printf("Using Standard Ridge Method")
			betas, err = calculateBetasUsingStandardRidge(Ytrn, Ztrn, lamlist)
		}
		if err != nil {
			return nil, nil, nil, err
		}

		// Prepare Ztst for prediction at time t
		Ztst := mat.NewVecDense(cols, nil)
		for c := 0; c < cols; c++ {
			Ztst.SetVec(c, Xstd.At(t, c)) // Fill Ztst with the t-th row of Xstd
		}

		// Scale Ztst by the same standard deviations
		scaleVecByStdDev(Ztst, stdDevs)

		// Predict for the next time step and optionally add back the means if data was demeaned
		// betas is a *mat.Dense where each column represents the beta coefficients for a lambda value
		for l := 0; l < nLambda; l++ {

			//Ztst := mat.NewVecDense(len(Ztrn.RawRowView(t)), Ztrn.RawRowView(t)) // Convert the test row to a VecDense
			//fmt.Printf("Ztst: %v", Ztst)
			betaVec := betas.ColView(l) // This returns a VecDense which implements mat.Vector

			pred := mat.Dot(betaVec, Ztst)

			if demean {
				// If demeaning was applied, adjust the prediction by adding back the mean of Y
				pred += meanY
			}

			// Store the prediction and perform any necessary rescaling
			YPrd.Set(t, l, pred)
			if YScaleStd != nil {
				// Ensure to check if YScaleStd is not nil to avoid panic in case it wasn't initialized due to stdize being false
				YPrdRescaled.Set(t, l, pred*YScaleStd.At(t, 0))
			} else {
				YPrdRescaled.Set(t, l, pred) // If not standardizing, just copy over the prediction
			}
		}
	}

	return YPrd, YPrdRescaled, BNorm, nil
}

*/

// Task represents a unit of work for the rolling window ridge regression.
type Task struct {
	StartIndex int
	EndIndex   int
	T          int
}

// TaskResult holds the result of a single task.
type TaskResult struct {
	T            int
	YPrd         *mat.Dense
	YPrdRescaled *mat.Dense
	BNorm        *mat.Dense
	Err          error
}

// Worker function to process tasks.
func worker(tasks <-chan Task, results chan<- TaskResult, Xstd, Ystd, YScaleStd *mat.Dense, trnwin int, demean bool, lamlist []float64, useKellyMethod bool) {
	for task := range tasks {
		result := processTask(task, Xstd, Ystd, YScaleStd, trnwin, demean, lamlist, useKellyMethod)
		results <- result
	}
}

// processTask processes a single time step for all lambda values.
func processTask(task Task, Xstd, Ystd, YScaleStd *mat.Dense, trnwin int, demean bool, lamlist []float64, useKellyMethod bool) TaskResult {
	t := task.T
	_, cols := Xstd.Dims()
	nLambda := len(lamlist)

	start := max(0, t-trnwin) // Calculate the start index to keep the window size constant
	end := t                  // The end index is exclusive, so using t directly works as intended

	// Adjust Ztrn and Ytrn to contain only data within the rolling window
	// Create a copy of ZtrnSlice. Use Clone to make a copy that does not share underlying data with Xstd.
	//   otherwise, the standardization will change the original data matrix
	Ztrn := mat.DenseCopyOf(Xstd.Slice(start, end, 0, cols).(*mat.Dense))
	Ytrn := mat.DenseCopyOf(Ystd.Slice(start, end, 0, 1).(*mat.Dense))

	var meanY float64
	if demean {
		// Demean Ytrn and store the mean for adding back mean after prediction
		YtrnCol := mat.Col(nil, 0, Ytrn)
		meanY = demeanVector(YtrnCol)
		// Copy the demeaned data back to Ytrn
		for i := 0; i < len(YtrnCol); i++ {
			Ytrn.Set(i, 0, YtrnCol[i])
		}

		// Demean Ztrn
		_, cols := Ztrn.Dims()
		for c := 0; c < cols; c++ {
			ZtrnCol := mat.Col(nil, c, Ztrn)
			_ = demeanVector(ZtrnCol) // Demean in place; mean values are not directly used here
			for r := 0; r < len(ZtrnCol); r++ {
				Ztrn.Set(r, c, ZtrnCol[r])
			}
		}
	}

	// Compute standard deviation for each column in Ztrn
	stdDevs := computeStdDev(Ztrn)
	// Scale Ztrn by its column-wise standard deviations
	scaleByStdDev(Ztrn, stdDevs)

	// Check if Kelly's method should be used
	rowsZtrn, colsZtrn := Ztrn.Dims()
	var betas *mat.Dense
	//var err error
	if useKellyMethod && colsZtrn > rowsZtrn {
		//fmt.Printf("Using Kelly Method")
		betas, _ = calculateBetasUsingKellyMethod(Ytrn, Ztrn, lamlist)
	} else {
		//fmt.Printf("Using Standard Ridge Method")
		betas, _ = calculateBetasUsingStandardRidge(Ytrn, Ztrn, lamlist)
	}
	//	if err != nil {
	//		return nil // err
	//	}

	// Prepare Ztst for prediction at time t
	Ztst := mat.NewVecDense(cols, nil)
	for c := 0; c < cols; c++ {
		Ztst.SetVec(c, Xstd.At(t, c)) // Fill Ztst with the t-th row of Xstd
	}

	// Scale Ztst by the same standard deviations
	scaleVecByStdDev(Ztst, stdDevs)

	// Prepare the result matrices for this task.
	YPrd := mat.NewDense(1, len(lamlist), nil)
	YPrdRescaled := mat.NewDense(1, len(lamlist), nil)
	BNorm := mat.NewDense(1, len(lamlist), nil)

	// Process each lambda value.
	// Predict for the next time step and optionally add back the means if data was demeaned
	// betas is a *mat.Dense where each column represents the beta coefficients for a lambda value
	for l := 0; l < nLambda; l++ {

		//Ztst := mat.NewVecDense(len(Ztrn.RawRowView(t)), Ztrn.RawRowView(t)) // Convert the test row to a VecDense
		//fmt.Printf("Ztst: %v", Ztst)
		betaVec := betas.ColView(l) // This returns a VecDense which implements mat.Vector

		pred := mat.Dot(betaVec, Ztst)

		if demean {
			// If demeaning was applied, adjust the prediction by adding back the mean of Y
			pred += meanY
		}

		// Store the prediction and perform any necessary rescaling
		YPrd.Set(0, l, pred)
		if YScaleStd != nil {
			// Ensure to check if YScaleStd is not nil to avoid panic in case it wasn't initialized due to stdize being false
			YPrdRescaled.Set(0, l, pred*YScaleStd.At(t, 0))
		} else {
			YPrdRescaled.Set(0, l, pred) // If not standardizing, just copy over the prediction
		}
	}

	return TaskResult{T: t, YPrd: YPrd, YPrdRescaled: YPrdRescaled, BNorm: BNorm}
}

func performRollingWindowRidgeRegression(Xstd, Ystd, YScaleStd *mat.Dense, trnwin int, demean bool, lamlist []float64, useKellyMethod bool) (*mat.Dense, *mat.Dense, *mat.Dense, error) {
	T, _ := Ystd.Dims()
	tasks := make(chan Task, T-trnwin)
	results := make(chan TaskResult, T-trnwin)

	var wg sync.WaitGroup
	numWorkers := runtime.NumCPU() // multiprocessing causes issues here!
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			worker(tasks, results, Xstd, Ystd, YScaleStd, trnwin, demean, lamlist, useKellyMethod)
		}()
	}

	// Distribute tasks.
	for t := trnwin; t < T; t++ {
		tasks <- Task{T: t}
	}
	close(tasks)

	// Wait for all workers to finish.
	go func() {
		wg.Wait()
		close(results)
	}()

	// Initialize result matrices with appropriate dimensions
	nLambda := len(lamlist)                       // Assume lamlist is defined in your context
	YPrd := mat.NewDense(T, nLambda, nil)         // Matrix for unscaled predictions
	YPrdRescaled := mat.NewDense(T, nLambda, nil) // Matrix for rescaled predictions
	BNorm := mat.NewDense(T, nLambda, nil)        // Matrix for beta norms

	// Collect results as they arrive
	for r := range results {

		if r.Err != nil {
			// Handle the error, perhaps logging it or stopping further processing
			fmt.Printf("Error processing time step %d: %v\n", r.T, r.Err)
			continue
		}

		// r.T is the time step for these results
		// Copy the data from each TaskResult's matrices to the corresponding rows in the final matrices
		if r.YPrd != nil {
			row := r.YPrd.RawRowView(0) // Assuming each result is a single row
			YPrd.SetRow(r.T, row)       // Set the row for time step r.T in YPrd
		}
		if r.YPrdRescaled != nil {
			row := r.YPrdRescaled.RawRowView(0)
			YPrdRescaled.SetRow(r.T, row) // Set the row for time step r.T in YPrdRescaled
		}
		if r.BNorm != nil {
			row := r.BNorm.RawRowView(0)
			BNorm.SetRow(r.T, row) // Set the row for time step r.T in BNorm
		}
	}

	return YPrd, YPrdRescaled, BNorm, nil
}

// benchmarkSim computes OOS forecasts using benchmark data, applying Ridge regression with several shrinkage parameters.
// It includes an option to standardize and/or demean the predictor and response variables.
func benchmarkSim(X, Y *mat.Dense, trnwin int, stdize bool, demean bool) (*mat.Dense, *mat.Dense, *mat.Dense, error) {
	startTime := time.Now()

	// Generate lambda values for ridge regression
	lamlist := generateLambdaList(-3, 3, 1)

	Xstd, Ystd, YScaleStd, _ := preprocessData(X, Y, trnwin, stdize)

	YPrd, YPrdRescaled, BNorm, _ := performRollingWindowRidgeRegression(Xstd, Ystd, YScaleStd, trnwin, demean, lamlist, false)

	fmt.Printf("Total runtime: %v seconds\n", time.Since(startTime).Seconds())
	return YPrd, YPrdRescaled, BNorm, nil
}

func generateRandomWeights(numRawVariables int, maxP int, seed int64) *mat.Dense {
	//_, d := X.Dims()

	localRand := rand.New(rand.NewSource(uint64(seed)))

	// Create a mean vector with zeros
	mu := make([]float64, numRawVariables)

	// Create a covariance matrix as an identity matrix, since our variance is 1
	// This represents the standard deviation of 1 for each dimension
	sigma := mat.NewSymDense(numRawVariables, nil)
	for i := 0; i < numRawVariables; i++ {
		sigma.SetSym(i, i, 1) // Set variance to 1 on the diagonal
	}

	normalDist, ok := distmv.NewNormal(mu, sigma, localRand)
	if !ok {
		panic("error creating normal distribution")
	}

	W := mat.NewDense(maxP, numRawVariables, nil)

	// Generate random projection matrix W
	for i := 0; i < maxP; i++ {
		sample := normalDist.Rand(nil)
		W.SetRow(i, sample)
	}

	return W
}

// computeRFF computes the Random Fourier Features for the given dataset X.
// numFeatures controls the number of random Fourier features to compute.
func computeRFF(XTransposed, randomWeights *mat.Dense, gamma float64, numFeatures int) *mat.Dense {
	// Ensure numFeatures is even and does not exceed twice the number of rows in wtmp.
	_, d := randomWeights.Dims()

	// Adjust numFeatures to account for both cosine and sine features.
	// the integer operations automatically rounds down
	numFeaturePairs := numFeatures / 2

	// Use only the required number of rows from randomWeights for the specified number of features.
	randomWeightsSelected := randomWeights.Slice(0, numFeaturePairs, 0, d).(*mat.Dense)

	_, numTimePeriods := XTransposed.Dims()
	RFF := mat.NewDense(numTimePeriods, numFeaturePairs*2, nil)

	// create feature pairs based on the random weights and raw features
	for i := 0; i < numFeaturePairs; i++ {

		w := randomWeightsSelected.RowView(i)

		for j := 0; j < numTimePeriods; j++ {

			x := XTransposed.ColView(j)

			dot := mat.Dot(w, x)

			// Set the cosine and sine features in the corresponding positions.
			RFF.Set(j, i, math.Cos(gamma*dot))
			RFF.Set(j, i+numFeaturePairs, math.Sin(gamma*dot))

		}

	}

	return RFF
}

// getLogspace generates numbers on a log scale between start and end, inclusive, with a given number of points.
func getLogspace(start, end float64, numPoints int) []float64 {
	if numPoints < 2 {
		return []float64{start, end}
	}

	// Calculate the step on the log scale.
	logStart, logEnd := math.Log10(start), math.Log10(end)
	step := (logEnd - logStart) / float64(numPoints-1)

	logspace := make([]float64, numPoints)
	for i := range logspace {
		logspace[i] = math.Pow(10, logStart+step*float64(i))
	}

	return logspace
}

// getUniqueIntLogspace generates a list of unique integers on a log scale, ensuring a minimum step size of 2.
func getUniqueIntLogspace(maxP, numPoints int) []int {
	PStart, PEnd := 2, maxP

	// Generate numbers on a log scale.
	numbers := getLogspace(float64(PStart), float64(PEnd), numPoints)

	// Convert to integers with rounding and ensure uniqueness and minimum step size of 2.
	uniqueNumbers := make([]int, 0, len(numbers))
	uniqueNumbers = append(uniqueNumbers, int(numbers[0])) // Start with the first number.
	for _, num := range numbers[1:] {
		nextInt := max(uniqueNumbers[len(uniqueNumbers)-1]+2, int(math.Round(num)))
		// Check for uniqueness is handled by the step size condition and slice logic.
		uniqueNumbers = append(uniqueNumbers, nextInt)
	}

	// Remove numbers exceeding PEnd and ensure final list respects the maximum limit.
	for i := len(uniqueNumbers) - 1; i >= 0; i-- {
		if uniqueNumbers[i] > PEnd {
			uniqueNumbers = uniqueNumbers[:i]
		} else {
			break // Numbers are sorted, so we can stop once we find the first valid one.
		}
	}

	return uniqueNumbers
}

// simulates Ridge regression with RFF for various complexities and lambdas.
func RffRidgeSim(X, Y *mat.Dense, iSim int, maxP int, gamma float64, trnwin int, stdize bool, demean bool) ([]*mat.Dense, []*mat.Dense, []*mat.Dense, error) {
	startTime := time.Now()

	// Generate lambda values for ridge regression.
	lamlist := generateLambdaList(-3, 3, 1)

	// Generate the list of complexities (P values).
	Plist := getUniqueIntLogspace(maxP, 30)

	fmt.Println("maxP:", maxP)

	// Standardize X and Y.
	Xstd, Ystd, YScaleStd, err := preprocessData(X, Y, trnwin, stdize)
	if err != nil {
		return nil, nil, nil, err
	}

	numTimePeriods, numRawVariables := Xstd.Dims()

	fmt.Println("numTimePeriods:", numTimePeriods, "numRawVariables", numRawVariables)

	// Initialize slices of matrices to store results for each P.
	YPrd := make([]*mat.Dense, len(Plist))
	YPrdRescaled := make([]*mat.Dense, len(Plist))
	BNorm := make([]*mat.Dense, len(Plist))

	// Generate RFF for the maximum number in Plist using computeRFF.
	//RFF := computeRFF(Xstd, gamma, maxP, int64(iSim)) // maxP*2 because of sine and cosine.

	//rff_rows, rff_cols := RFF.Dims()

	// build transpose of X for calculation of RFFs
	XstdT := mat.DenseCopyOf(Xstd.T()) // Transposing matrix X

	randomWeights := generateRandomWeights(numRawVariables, maxP, int64(iSim)) // should be maxP rows x numRawVariables columns
	randomWeights_rows, randomWeights_cols := randomWeights.Dims()

	fmt.Println("randomWeights_rows:", randomWeights_rows, "randomWeights_cols", randomWeights_cols)

	for i, P := range Plist {

		fmt.Println("P:", P)

		// Calculate the random fourier features for this iteration of P
		RFF := computeRFF(XstdT, randomWeights, gamma, P)

		// Perform Ridge regression using the selected RFF features.
		YPrdTemp, YPrdRescaledTemp, BNormTemp, err := performRollingWindowRidgeRegression(RFF, Ystd, YScaleStd, trnwin, demean, lamlist, true)
		if err != nil {
			return nil, nil, nil, err
		}

		// Store the results for the current P value.
		YPrd[i] = YPrdTemp
		YPrdRescaled[i] = YPrdRescaledTemp
		BNorm[i] = BNormTemp
	}

	fmt.Printf("Total runtime: %v seconds\n", time.Since(startTime).Seconds())

	return YPrd, YPrdRescaled, BNorm, nil
}

/*

// RFFSim with multiprocessing:
func RffRidgeSim(X, Y *mat.Dense, iSim int, maxP int, gamma float64, trnwin int, stdize bool, demean bool) ([]*mat.Dense, []*mat.Dense, []*mat.Dense, error) {
	startTime := time.Now()

	// Use all available CPUs
	runtime.GOMAXPROCS(runtime.NumCPU())

	// Generate lambda values for ridge regression.
	lamlist := generateLambdaList(-3, 3, 1)

	// Generate the list of complexities (P values).
	Plist := getUniqueIntLogspace(maxP, 30)

	fmt.Println("maxP:", maxP)

	// Standardize X and Y.
	Xstd, Ystd, YScaleStd, err := preprocessData(X, Y, trnwin, stdize)
	if err != nil {
		return nil, nil, nil, err
	}

	numTimePeriods, numRawVariables := Xstd.Dims()

	fmt.Println("numTimePeriods:", numTimePeriods, "numRawVariables", numRawVariables)

	// build transpose of X for calculation of RFFs
	XstdT := mat.DenseCopyOf(Xstd.T()) // Transposing matrix X

	randomWeights := generateRandomWeights(numRawVariables, maxP, int64(iSim)) // should be maxP rows x numRawVariables columns

	var wg sync.WaitGroup
	var mu sync.Mutex // For safely updating results in multiprocessing loop

	// Initialize slices of matrices to store results for each P.
	YPrd := make([]*mat.Dense, len(Plist))
	YPrdRescaled := make([]*mat.Dense, len(Plist))
	BNorm := make([]*mat.Dense, len(Plist))

	for i, P := range Plist {
		wg.Add(1)
		go func(i int, P int) {
			defer wg.Done()

			RFF := computeRFF(XstdT, randomWeights, gamma, P)

			YPrdTemp, YPrdRescaledTemp, BNormTemp, err := performRollingWindowRidgeRegression(RFF, Ystd, YScaleStd, trnwin, demean, lamlist, true)
			if err != nil {
				fmt.Println("Error in computation:", err)
				return
			}

			mu.Lock()
			YPrd[i] = YPrdTemp
			YPrdRescaled[i] = YPrdRescaledTemp
			BNorm[i] = BNormTemp
			mu.Unlock()
		}(i, P)
	}

	// Wait for all goroutines to finish
	wg.Wait()

	fmt.Printf("Total runtime: %v seconds\n", time.Since(startTime).Seconds())
	return YPrd, YPrdRescaled, BNorm, nil
}

*/
