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
	"math"
	"math/cmplx"
	shmeh "github.com/sillsm/shmensor/shmensor"

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
	for in, ch := range signature {
		switch string(ch) {
		case "u":
			yi *= dim[in]
			if (y)%yi == 0 {
				retY++
			}
		case "d":
			xi *= dim[in]
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

			fmt.Printf("%.5f\t", grid[y][x])

		}
		fmt.Printf("\n")
	}
}

func E(e ...shmeh.Expression) []shmeh.Expression {
	return e
}

func main() {
	table := []struct {
		t []shmeh.Expression
		// Reshape it to make it visually compelling.
		// In reality, all these tensors should be all "uuuu", or the
		// contraction with the "ud" partial derivative doesn't make sense.
		reshape        string
		aTrans, bTrans int
		desc           string
	}{
		{E(newDFTTensor(2).U("i").D("j")),
			"ud",
			0, 0,
			"Let's start with visualizing the components of some DFT Matrices. 2:"},
		{E(newDFTTensor(3).U("i").D("j")),
			"ud",
			0, 0,
			"3:"},
		{E(newDFTTensor(4).U("i").D("j")),
			"ud",
			0, 0,
			"4:"},
		{E(newDFTTensor(5).U("i").D("j")),
			"ud",
			0, 0,
			"5:"},
		{E(newDFTTensor(8).U("i").D("j")),
			"ud",
			0, 0,
			"8:"},
		{E(newDFTTensor(5).U("i").D("j"), newIDFTTensor(5).U("j").D("k")),
			"ud",
			0, 0,
			"Let's demonstrate the IDFT matrix and the DFT matrix are inverses."},
		{E(
			newIDFTTensor(9).U("z").D("h"),           // Finally, we invert.
			newComplexDirac3(9).U("h").D("f").D("g"), // Hadamard product those babies.
			newDFTTensor(9).U("f").D("a"),            // DFT the first two polynomials.
			newDFTTensor(9).U("g").D("x"),
			Embed(5, 9).U("a").D("b"), newVec(5, 4, 3, 2, 1).U("b"), //new vector index is a
			Embed(5, 9).U("x").D("y"), newVec(5, 6, 7, 8, 9).U("y")), //new vector index is x
			"u",
			0, 0,
			"Finally, we compute the product of the DFT of both polynomials, then invert."},
	}
	for _, elt := range table {
		tensor, err, profiler := shmeh.Eval(elt.t...)
		// Transpose?
		if elt.aTrans != 0 || elt.bTrans != 0 {
			tensor, err = shmeh.Transpose(tensor, elt.aTrans, elt.bTrans)

		}

		tensor.Reshape(elt.reshape)

		if err != nil {
			panic(err)
		}
		fmt.Printf("%v\n", elt.desc)
		//fmt.Printf("%v", tensor)

		VisualizePolynomial(tensor, nil, nil)
		fmt.Printf("%v\n", profiler)
		fmt.Printf("\n\n\n")
	}

}

// 3x^2 + 5x + 10
var P1 = shmeh.NewIntTensor(
	func(i ...int) int {
		z := []int{3, 5, 10}
		return z[i[0]]
	},
	"u",
	[]int{3},
)

// x^2y^2 + 3x^2y + x^2 + 5xy^2 + 4xy + y^2 + 2
var P2 = shmeh.NewIntTensor(
	func(i ...int) int {
		z := [][]int{
			//y^2 y  c
			{1, 3, 1}, // x^2
			{5, 4, 0}, // x
			{1, 0, 2}} // c
		return z[i[0]][i[1]]
	},
	"ud",
	[]int{3, 3},
)

var elements = shmeh.NewIntTensor(
	func(i ...int) int {
		z := [][]int{
			//y^2 y  c
			{1, 2, 3}, // x^2
			{4, 5, 6}, // x
			{7, 8, 9}} // c
		return z[i[0]][i[1]]
	},
	"ud",
	[]int{3, 3},
)

// 3x^2y^2z^2 + xyz^2 + 2yz^2 + 5x^2y^2z + 3x^2z + 7xy
var P3 = shmeh.NewIntTensor(
	func(i ...int) int {
		z := [][][]int{

			//y^2 y  c
			{ //z^2
				{3, 0, 0},  // x^2
				{7, 1, 0},  // x
				{0, 2, 0}}, // c
			//y^2 y  c
			{ // z
				{5, 0, 3},  // x^2
				{0, 0, 0},  // x
				{0, 0, 0}}, // c
			//y^2 y  c
			{ // c
				{0, 0, 0},  // x^2
				{0, 7, 0},  // x
				{0, 0, 0}}, // c
		}
		return z[i[0]][i[1]][i[2]]
	},
	"uuu",
	[]int{3, 3, 3},
)

// Derivative on 2nd degree polynomials.
var D3 = shmeh.NewIntTensor(
	func(i ...int) int {
		z := [][]int{
			{0, 0, 0},
			{2, 0, 0},
			{0, 1, 0}}
		return z[i[0]][i[1]]
	},
	"ud",
	[]int{3, 3},
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

var dirac_delta3 = shmeh.NewIntTensor(
	identity,
	"uud",
	[]int{3, 3, 3},
)

var dirac_delta5 = shmeh.NewIntTensor(
	identity,
	"uud",
	[]int{5, 5, 5},
)

var dirac_delta9 = shmeh.NewIntTensor(
	identity,
	"uud",
	[]int{9, 9, 9},
)

// Shift the 0th row 0 to the right, 1st row 1 to the right
// Shift the nth row n to the right.
var ProgressiveShift3 = shmeh.NewIntTensor(
	func(i ...int) int {
		z := [][][]int{
			{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1}}, // 1/n!
			{
				{0, 0, 0},
				{1, 0, 0},
				{0, 1, 0}},
			{
				{0, 0, 0},
				{0, 0, 0},
				{1, 0, 0}},
		}
		return z[i[0]][i[1]][i[2]]
	},
	"dud",
	[]int{3, 3, 3},
)

// Arrayed on the first index
func ProgressiveShift(i ...int) int {
	if len(i) != 3 {
		panic("Trying call progressive shift != 3 entries.")
	}
	x, y := i[1], i[2]
	z := i[0]
	if (y + z) == x {
		return 1
	}
	return 0
}

var ProgressiveShift5 = shmeh.NewIntTensor(
	ProgressiveShift,
	"dud",
	[]int{5, 5, 5},
)

var ProgressiveShift9 = shmeh.NewIntTensor(
	ProgressiveShift,
	"dud",
	[]int{9, 9, 9},
)

// Embed returns a matrix which will
// embed an input a vector in a different vector space.
// When larger, it zero-pads all new dimensions.
func Embed(inputDim, outputDim int) *shmeh.Tensor {
	f := func(i ...int) complex128 {
		return complex(float64(identity(i...)), 0)
	}
	t := shmeh.NewComplexTensor(
		f,
		"ud",
		[]int{outputDim, inputDim},
	)
	return &t
}

// Does a DFT matrix.
func DFT(n int) func(i ...int) complex128 {
	increment := -2 * math.Pi / float64(n)
	var motherRow []complex128
	for j := 0; j < n; j++ {
		num := increment * float64(j)
		x, y := math.Cos(num), math.Sin(num)
		if math.Abs(x) < .000000000001 {
			x = 0
		}
		if math.Abs(y) < .000000000001 {
			y = 0
		}
		motherRow = append(motherRow, complex(x, y))
	}

	f := func(i ...int) complex128 {
		if len(i) != 2 {
			panic("Trying to get DFT coefficients where dim != 2")
		}
		x, y := i[0], i[1]
		return motherRow[(x * y % n)]
	}
	return f
}

func newComplexDirac3(size int) *shmeh.Tensor {
	f := func(i ...int) complex128 {
		if i[0] == i[1] && i[1] == i[2] {
			return complex(1., 0.)
		}
		return complex(0., 0.)
	}

	t := shmeh.NewComplexTensor(
		f,
		"dud",
		[]int{size, size, size})
	return &t
}

func newDFTTensor(size int) *shmeh.Tensor {
	t := shmeh.NewComplexTensor(
		DFT(size),
		"ud",
		[]int{size, size})
	return &t

}

// Inverse DFT is equal to the conjugate transpose of the DFT.
// Because a DFT is a self-adjoint unitary matrix.
// Or something?
func newIDFTTensor(size int) *shmeh.Tensor {
	dft := DFT(size)
	f := func(i ...int) complex128 {
		return cmplx.Conj(dft(i...)) / complex(float64(size), 0) // conjugate that DFT, then norm it by the size.
	}
	t := shmeh.NewComplexTensor(
		f,
		"ud",
		[]int{size, size})
	return &t
}

// New vector helper function.
func newVec(i ...int) *shmeh.Tensor {
	t := shmeh.NewComplexTensor(
		func(j ...int) complex128 {
			return complex(float64(i[j[0]]), 0)
		},
		"u",
		[]int{len(i)})
	return &t
}
