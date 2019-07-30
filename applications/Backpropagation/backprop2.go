// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Equations from http://neuralnetworksanddeeplearning.com/chap2.html
// Specific example weights from https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
package main

import (
	"fmt"
	shmeh "github.com/sillsm/shmensor/shmensor"
	"math"
)

func forwardPass(weights, activation, biases shmeh.Tensor) shmeh.Tensor {
	rightShift := newMatrix(
		[]float64{0, 1, 0},
		[]float64{0, 0, 1},
		[]float64{0, 0, 0},
	)
	sigmoid := shmeh.NewRealFunction(func(r float64) float64 {
		return math.Exp(r) / (1. + math.Exp(r))
	})
	expression :=
		shmeh.Apply{
			shmeh.NewRealScalar(1.),
			shmeh.Apply{
				sigmoid, // Sigmoid activation function.
				shmeh.Plus{
					biases, // Add biases.
					shmeh.E(
						weights.D("a").U("b").D("c"),
						activation.U("c").D("d"),
						dirac3(3, 3, 2).U("d").D("e").U("a"), // Multiply weight matrix i by activation column i
						rightShift.U("e").D("f"),             // Move signal forward one column.
					)},
			},
		}

	s, err, _ := expression.Eval()
	if err != nil {
		panic(err)
	}
	return s
}

func backwardPass(weights, errors, biases shmeh.Tensor) shmeh.Tensor {
	rightShift := newMatrix(
		[]float64{0, 1, 0},
		[]float64{0, 0, 1},
		[]float64{0, 0, 0},
	)
	sigmoid := shmeh.NewRealFunction(func(r float64) float64 {
		return math.Exp(r) / (1. + math.Exp(r))
	})
	expression :=
		shmeh.Apply{
			shmeh.NewRealScalar(1.),
			shmeh.Apply{
				sigmoid, // Sigmoid activation function.
				shmeh.Plus{
					biases, // Add biases.
					shmeh.E(
						weights.D("a").U("b").D("c"),
						errors.U("c").D("d"),
						dirac3(3, 3, 2).U("d").D("e").U("a"), // Multiply weight matrix i by activation column i
						rightShift.U("e").D("f"),             // Move signal forward one column.
					)},
			},
		}

	s, err, _ := expression.Eval()
	if err != nil {
		panic(err)
	}
	return s
}

func delSigmaZ(activation, biases shmeh.Tensor) shmeh.Tensor {
	// Left shift.
	rightShift := newMatrix(
		[]float64{0, 1, 0},
		[]float64{0, 0, 1},
		[]float64{0, 0, 0},
	)
	delSigmoid := shmeh.NewRealFunction(func(r float64) float64 {
		sigmoid := math.Exp(r) / (1. + math.Exp(r))
		return sigmoid * (1 - sigmoid)
	})
	expression :=
		shmeh.Apply{
			delSigmoid, // DelSigmoid function.
			shmeh.Plus{
				biases, // Add biases.
				shmeh.E(
					weights.D("a").U("b").D("c"),
					activation.U("c").D("d"),
					dirac3(3, 3, 2).U("d").D("e").U("a"), // Multiply weight matrix i by activation column i
					rightShift.U("e").D("f"),             // Move signal forward one column.
				)},
		}

	s, err, _ := expression.Eval()
	if err != nil {
		panic(err)
	}
	return s
}

func totalError(output, target shmeh.Tensor) float64 {
	if len(output.Dimension()) > 1 {
		fmt.Printf("Error %v", output)
		panic("Tried to get error of poor fit tensor")
	}
	if len(target.Dimension()) > 1 {
		fmt.Printf("Error %v", output)
		panic("Tried to get error of poor fit tensor")
	}
	if target.Dimension()[0] != output.Dimension()[0] {
		panic("Target and output layer don't match up.")
	}
	out := output.Reify()
	tar := target.Reify()
	totalErr := 0.
	for i := range out {
		t := tar[i][0].(float64)
		o := out[i][0].(float64)
		totalErr += (t - o) * (t - o) / 2.

	}
	return totalErr
}

func outputLayerError(output, target shmeh.Tensor) []float64 {
	var ret []float64
	o := output.Reify()
	t := target.Reify()
	for i := range o {
		ret = append(ret, o[i][0].(float64)-t[i][0].(float64))
	}
	return ret
}

// Need to write checks here.
func setLeftColumn(left []float64, t shmeh.Tensor) shmeh.Tensor {
	if len(t.Signature()) != 2 {
		panic("Trying to set a column of a non-matrix.")
	}
	a := t.Reify()
	ten := shmeh.NewRealTensor(
		func(j ...int) float64 {
			if j[1] == 0 { // if we're trying to access the left column
				return left[j[0]]
			}
			return a[j[0]][j[1]].(float64)
		},
		"ud",
		t.Dimension())
	return ten
}

// Need to write checks here.
func setRightColumn(left []float64, t shmeh.Tensor) shmeh.Tensor {
	if len(t.Signature()) != 2 {
		panic("Trying to set a column of a non-matrix.")
	}
	a := t.Reify()
	ten := shmeh.NewRealTensor(
		func(j ...int) float64 {
			if j[1] == t.Dimension()[1]-1 { // if we're trying to access the left column
				return left[j[0]]
			}
			return a[j[0]][j[1]].(float64)
		},
		"ud",
		t.Dimension())
	return ten
}

func main() {
	// Now run a [1,1,0] two input vector into the system
	activation := newMatrix(
		[]float64{.05, 0, 0},
		[]float64{.1, 0, 0},
	)
	targetOutput := newVec(.01, .99)
	targetOutput = targetOutput
	biases := newMatrix(
		[]float64{0, .35, .60},
		[]float64{0, .35, .60},
	)

	fmt.Printf("Current Weights %v", weights)

	activation = forwardPass(weights, activation, biases)
	activation = setLeftColumn([]float64{.05, .1}, activation)
	fmt.Printf("First Pass %v", activation)

	activation = forwardPass(weights, activation, biases)
	activation = setLeftColumn([]float64{.05, .1}, activation)
	fmt.Printf("Second Pass %v", activation)

	lastColumn := shmeh.E(activation.U("x").D("y"), newVec(0, 0, 1).U("y"))
	output, err, _ := lastColumn.Eval()
	if err != nil {
		panic(err)
	}
	fmt.Printf("Output layer%v", output)
	fmt.Printf("\nTotal Error: %v\n\n", totalError(*targetOutput, output))
	fmt.Printf("Target Output %v", targetOutput)

	dSigmaWRTActivations := delSigmaZ(activation, biases)

	fmt.Printf("Del Sigma %v", dSigmaWRTActivations)
	errorLayer := newMatrix(
		[]float64{0, 0, 0},
		[]float64{0, 0, 0},
	)

	errorLayer = setRightColumn(outputLayerError(output, *targetOutput), errorLayer)
	fmt.Printf("Err %v", errorLayer)

	return

	fmt.Printf("Now we're going to use a simple tensor expression\n" +
		"and repeat its evaluation to drive a signal\n" +
		"from the input to the output of a neural net.\n\nThe expression takes the nth " +
		"weight matrix above, \napplies it to the nth column in the\n" +
		"activation matrix, and then shifts everything to the right.\n")
	fmt.Printf("%v", activation)

	// Do forward pass.

	fmt.Printf("This is almost the correct expression.\n" +
		"We also need to add biases to each node\n" +
		"and then apply a sigmoid function.\n")

	fmt.Printf("\nChanging gears, let's say we had an error quantity " +
		"for the output layer.\nHow might we propogate it backwards?\n")

	tWeights, _ := shmeh.Transpose(weights, 1, 2)
	//VisualizePolynomial(tWeights, nil, nil)

	leftShift := newMatrix(
		[]float64{0, 0, 0, 0},
		[]float64{1, 0, 0, 0},
		[]float64{0, 1, 0, 0},
		[]float64{0, 0, 1, 0},
	)
	errorInOutput := newMatrix(
		[]float64{0, 0, 0, 15},
		[]float64{0, 0, 0, 0},
		[]float64{0, 0, 0, 0},
	)

	pad := newMatrix(
		[]float64{0, 0, 0},
		[]float64{1, 0, 0},
		[]float64{0, 1, 0},
		[]float64{0, 0, 1},
	)

	pad.Reshape("du")

	backProp1 := shmeh.E(
		pad.D("a").U("z"),
		tWeights.D("z").U("b").D("c"),
		errorInOutput.U("c").D("d"),
		dirac3(4, 4, 4).U("d").D("e").U("a"), // Used to tie the indices of the weights and activations.
		leftShift.U("e").D("f"),              // Right-shift.
	)
	fmt.Printf("\nError %v\n", errorInOutput)
	for i := 0; i < 3; i++ {
		fmt.Printf("%v application", i)
		errorInOutput, _, _ = backProp1.Eval()
		fmt.Printf("%v", errorInOutput)
	}
	fmt.Printf("\nFinally, we take the back-propagated errors and the\n" +
		"forward propagated activations, and munge them together.\n")
	activations := newMatrix(
		[]float64{.1, .4, .7, .54},
		[]float64{.2, .5, .8, 0},
		[]float64{.0, .6, .9, 0},
	)

	errors := newMatrix(
		[]float64{1, 3, 6, 15},
		[]float64{2, 4, 7, 0},
		[]float64{0, 5, 8, 0},
	)
	activations = activations
	errors = errors

	delErrorWrtWeights := shmeh.E(
		dirac3(3, 4, 4).U("x").D("b").D("c"),
		activations.U("a").D("b"),
		errors.U("d").D("c"),
	)
	fmt.Printf("Current Weights\n")
	//VisualizePolynomial(weights, nil, nil)
	fmt.Printf("Activations %v", activations)
	fmt.Printf("Errors %v", errors)

	d, _, _ := delErrorWrtWeights.Eval()
	d.Reshape("dud")
	//VisualizePolynomial(d, nil, nil)
}

var weights = shmeh.NewRealTensor(
	func(i ...int) float64 {
		z := [][][]float64{
			{
				{.15, .20},
				{.25, .30}},
			{
				{.40, .45},
				{.50, .55}},
		}
		return z[i[0]][i[1]][i[2]]
	},
	"dud",
	// UD = (u)(ud)
	[]int{2, 2, 2},
)

func identity(i ...int) int {
	val := i[0]
	for _, elt := range i {
		if elt != val {
			return 0
		}
	}
	return 1
}

func identityFloat(i ...int) float64 {
	val := i[0]
	for _, elt := range i {
		if elt != val {
			return 0
		}
	}
	return 1
}

var dirac_delta4 = shmeh.NewRealTensor(
	identityFloat,
	"udu",
	[]int{4, 4, 3},
)

func dirac3(x, y, z int) *shmeh.Tensor {
	t := shmeh.NewRealTensor(
		identityFloat,
		"udu",
		[]int{x, y, z},
	)
	return &t
}

// Embed returns a matrix which will
// embed an input a vector in a different vector space.
// When larger, it zero-pads all new dimensions.
func Embed(inputDim, outputDim int) *shmeh.Tensor {
	t := shmeh.NewRealTensor(
		identityFloat,
		"ud",
		[]int{outputDim, inputDim},
	)
	return &t
}

// New vector helper function.
func newVec(i ...float64) *shmeh.Tensor {
	t := shmeh.NewRealTensor(
		func(j ...int) float64 {
			return i[j[0]]
		},
		"u",
		[]int{len(i)})
	return &t
}

// New matrix helper function.
func newMatrix(f ...[]float64) shmeh.Tensor {
	t := shmeh.NewRealTensor(
		func(j ...int) float64 {
			return f[j[0]][j[1]]
		},
		"ud",
		[]int{len(f), len(f[0])})
	return t
}
