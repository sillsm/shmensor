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
package main

import (
	"fmt"
	shmeh "github.com/sillsm/shmensor/shmensor"
	"math"
)

// Prettified
// u
// u
// u d d d

func detectNesting(signature string, dim []int, x, y int) (int, int) {
	retX := 0
	retY := 0
	xi := 1
	yi := 1
	for i := len(signature) - 1; i >= 0; i-- {
		ch := signature[i]
		//for in, ch := range signature {
		switch string(ch) {
		case "u":
			yi *= dim[i]
			if (y)%yi == 0 {
				retY++
			}
		case "d":
			xi *= dim[i]
			if (x)%xi == 0 {
				retX++
			}
		}
	}
	return retX, retY
}

// This function prints the nested covariant labels for a tensor.
func printCovariantLabels(d []int) {
	names := []string{"z", "y"}
	dim := 1
	for _, elt := range d {
		dim *= elt
	}
	tally := dim
	for vari, arity := range d {
		tally /= arity
		fmt.Println()
		ch := -1
		for i := 0; i < dim; i++ {
			j := (i / tally) % arity
			if j != ch && (i%tally) == tally/2 {
				if len(d) > len(names) {
					panic("Didn't plan more than 2 covariant indices.")
				}
				fmt.Printf("%v", names[len(names)-len(d):][vari])
				fmt.Printf("%v\t", arity-j-1)
				ch = j
				continue
			}
			fmt.Printf("\t")
		}
	}
}

func VisualizePolynomial(t shmeh.Tensor, contraLabels, coLabels [][]string) {
	grid := t.Reify()
	// Label the top
	// the first multiple of 3 (the zs)
	// then
	// the second multuple of 3 (the ys)

	printCovariantLabels(t.CovariantIndices())

	for y, row := range grid {
		_, lines := detectNesting(t.Signature(), t.Dimension(), 0, y)
		for i := 0; i < lines; i++ {
			fmt.Printf("\n")
			for j := 0; j < len(row); j++ {
				fmt.Printf("######\t")
			}
			fmt.Printf("\n")
		}
		for x := range row {
			bars, _ := detectNesting(t.Signature(), t.Dimension(), x, 0)
			for i := 0; i < bars; i++ {
				fmt.Printf("|")
			}

			fmt.Printf("%.3f\t", grid[y][x])

		}
		fmt.Printf("\n")
	}
}

func forwardPass(weights, activation, biases shmeh.Tensor) shmeh.Tensor {
	/*rightShift := newMatrix(
	[]float64{0, 1, 0, 0},
	[]float64{0, 0, 1, 0},
	[]float64{0, 0, 0, 1},
	[]float64{0, 0, 0, 0},
	)*/
	expression := shmeh.E(
		weights.D("a").U("b").D("c"),
		activation.U("c").D("d"),
		dirac3(3, 3, 2).U("d").D("e").U("a"), // Used to tie the indices of the
	)

	a, err, _ := expression.Eval()
	if err != nil {
		panic(err)
	}
	p, err := shmeh.Plus(biases, a)
	if err != nil {
		panic(err)
	}

	sigmoid := shmeh.NewRealFunction(func(r float64) float64 {
		return math.Exp(r) / (1. + math.Exp(r))
	})
	s, err := shmeh.Apply(sigmoid, p)
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
	biases := newMatrix(
		[]float64{.35, .60, 0},
		[]float64{.35, .60, 0},
	)

	t := forwardPass(weights, activation, biases)

	fmt.Printf("%v", t)
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
	VisualizePolynomial(tWeights, nil, nil)

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
	VisualizePolynomial(weights, nil, nil)
	fmt.Printf("Activations %v", activations)
	fmt.Printf("Errors %v", errors)

	d, _, _ := delErrorWrtWeights.Eval()
	d.Reshape("dud")
	VisualizePolynomial(d, nil, nil)
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
