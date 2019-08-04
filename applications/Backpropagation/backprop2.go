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

func weightedInput(weights, activation, biases shmeh.Tensor) shmeh.Tensor {
	rightShift := newMatrix(
		[]float64{0, 1, 0},
		[]float64{0, 0, 1},
		[]float64{0, 0, 0},
	)
	expression := shmeh.Plus{
		biases, // Add biases.
		shmeh.E(
			weights.D("a").U("b").D("c"),
			activation.U("c").D("d"),
			dirac3(3, 3, 2).U("d").D("e").U("a"), // Multiply weight matrix i by activation column i
			rightShift.U("e").D("f"),             // Move signal forward one column.
		)}

	s, err, _ := expression.Eval()
	if err != nil {
		panic(err)
	}
	return s
}

func forwardPass(weights, activation, biases shmeh.Tensor) shmeh.Tensor {
	sigmoid := shmeh.NewRealFunction(func(r float64) float64 {
		return math.Exp(r) / (1. + math.Exp(r))
	})
	expression :=
		shmeh.Apply{
			sigmoid, // Sigmoid activation function.
			weightedInput(weights, activation, biases),
		}

	s, err, _ := expression.Eval()
	if err != nil {
		panic(err)
	}
	return s
}

func backwardPass(weights, errors, delSigmaZ shmeh.Tensor) shmeh.Tensor {
	leftShift := newMatrix(
		[]float64{0, 0},
		[]float64{1, 0},
	)
	leftShift = leftShift
	// transpose weights
	tWeights, err := shmeh.Transpose(weights, 1, 2)
	if err != nil {
		panic(err)
	}

	expression := shmeh.E(
		tWeights.D("a").U("b").D("c"),
		errors.U("c").D("d"),
		dirac3(2, 2, 2).U("d").D("e").U("a"), // Multiply weight matrix i by activation column i
		leftShift.U("e").D("f"),              // Move signal backward one column.
		// Finish with a Hadamard product of this U(b).D(f) with delSigma
		delSigmaZ.U("w").D("x"),
		dirac3(2, 2, 2).U("w").D("y").U("b"),
		dirac3(2, 2, 2).U("x").D("z").U("f"),
	)

	s, err, _ := expression.Eval()
	if err != nil {
		panic(err)
	}
	s.Reshape("ud")
	return s
}

func delSigmaZ(activation, biases shmeh.Tensor) shmeh.Tensor {
	delSigmoid := shmeh.NewRealFunction(func(r float64) float64 {
		sigmoid := math.Exp(r) / (1. + math.Exp(r))
		return sigmoid * (1 - sigmoid)
	})
	expression :=
		shmeh.Apply{
			delSigmoid, // DelSigmoid function.
			weightedInput(weights, activation, biases),
		}

	s, err, _ := expression.Eval()
	if err != nil {
		panic(err)
	}

	// Finish by cutting off the input column.
	// No weighted activation there.
	cutLeftColumn := newMatrix( //mix of cut and shift-left.
		[]float64{0, 0},
		[]float64{1, 0},
		[]float64{0, 1},
	)

	cut := shmeh.E(
		s.U("a").D("b"),
		cutLeftColumn.U("b").D("c"),
	)
	s, err, _ = cut.Eval()
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

func outputLayerError(output, target, weightedActivations shmeh.Tensor) []float64 {
	var ret []float64
	activations := weightedActivations.Reify()
	o := output.Reify()
	t := target.Reify()
	for i := range o {
		a := activations[i][len(activations[i])-1].(float64)
		ret = append(ret, (o[i][0].(float64)-t[i][0].(float64))*a)
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

func getUpdatedWeights(errors, activations shmeh.Tensor) shmeh.Tensor {

	// Cut the input column from activations.
	// No weighted activation there.
	cutLeftColumn := newMatrix( //mix of cut and shift-left.
		[]float64{1, 0},
		[]float64{0, 1},
		[]float64{0, 0},
	)

	cut := shmeh.E(
		activations.U("a").D("b"),
		cutLeftColumn.U("b").D("c"),
	)
	activations, err, _ := cut.Eval()
	if err != nil {
		panic(err)
	}
	expression := shmeh.E(
		dirac3(2, 2, 2).U("x").D("b").D("c"),
		activations.U("a").D("b"),
		errors.U("d").D("c"),
	)

	s, err, _ := expression.Eval()
	if err != nil {
		panic(err)
	}
	s.Reshape("dud")
	return s
}

func updateWeights(oldWeights, newWeights shmeh.Tensor, learningRate float64) shmeh.Tensor {
	// transpose weights
	// TODO(Xam: investigate this last transpose here.)
	// Why was this necessary to get results to agree with mazur?
	newWeights, err := shmeh.Transpose(newWeights, 1, 2)
	if err != nil {
		panic(err)
	}
	// Investigate further.

	expression :=
		shmeh.Plus{
			oldWeights,
			shmeh.Apply{
				shmeh.NewRealScalar(-1. * learningRate),
				newWeights,
			},
		}
	s, err, _ := expression.Eval()
	if err != nil {
		panic(err)
	}
	return s
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
	// Notice we clip the input column; no error there.
	errorLayer := newMatrix(
		[]float64{0, 0},
		[]float64{0, 0},
	)

	errorLayer = setRightColumn(outputLayerError(output, *targetOutput, dSigmaWRTActivations), errorLayer)
	fmt.Printf("Err output layer%v", errorLayer)
	errorLayer = backwardPass(weights, errorLayer, dSigmaWRTActivations)
	errorLayer = setRightColumn(outputLayerError(output, *targetOutput, dSigmaWRTActivations), errorLayer)
	fmt.Printf("Err 1 pass back %v", errorLayer)
	fmt.Printf("Activations %v", activation)

	newWeights := getUpdatedWeights(errorLayer, activation)
	fmt.Printf("Del Weights %v", newWeights)

	fmt.Printf("Initial Weights %v", weights)

	learningRate := .5
	weights = updateWeights(weights, newWeights, learningRate)
	fmt.Printf("Final updated weights %v", weights)

	return
	/*
		To compare: Final updated weights
		[0.1497807161327628 0.1997511436323696 0.35891647971788465 0.46130127023873757]
		[0.24956143226552566 0.29950228726473915 0.4586661860762334 0.5613701211079891]
	*/

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

func identityFloat(i ...int) float64 {
	val := i[0]
	for _, elt := range i {
		if elt != val {
			return 0
		}
	}
	return 1
}

func dirac3(x, y, z int) *shmeh.Tensor {
	t := shmeh.NewRealTensor(
		identityFloat,
		"udu",
		[]int{x, y, z},
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
