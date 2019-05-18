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
	shmeh "shmensor/shmensor"
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

			fmt.Printf("%v\t", grid[y][x])

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
		{E(D3.U("i").D("j"), P1.U("j")),
			"u",
			0, 0,
			"Partial derivative of 3x^2 + 5x + 10."},
		{E(P2.U("m").U("k")),
			"ud",
			0, 0,
			"Matrix representation of polynomial\nx^2y^2 + 3x^2y + x^2 + 5xy^2 + 4xy + y^2 + 2"},
		{E(D3.U("i").D("m"), P2.U("m").U("k")),
			"ud",
			0, 0,
			"Partial derivative w.r.t x"},
		{E(D3.U("i").D("k"), P2.U("m").U("k")),
			"ud",
			0, 1,
			"Partial derivative w.r.t y"},
		{E(P3.U("l").U("j").U("m")),
			"dud",
			0, 0,
			"Matrix representation of polynomial\n3x^2y^2z^2 + xyz^2 + 2yz^2 + 5x^2y^2z + 3x^2z + 7xy"},
		{E(D3.U("i").D("l"), P3.U("l").U("j").U("m")),
			"dud",
			0, 0,
			"Derivative w.r.t. z"},
		{E(D3.U("i").D("j"), P3.U("l").U("j").U("m")),
			"udd",
			0, 0,
			"Derivative w.r.t. x"},
		{E(D3.U("i").D("z"), P3.U("l").U("j").U("z")),
			"udd",
			0, 2,
			"Derivative w.r.t. y "},
		//
		// We finish with some Taylor shifts!
		//
		{E(DerivativeTower2.U("i").D("j").U("k")),
			"dud",
			0, 0,
			"Tower of all nth derivatives of quadratic polynomials."},
		{E(DerivativeTower2.D("i").U("j").D("k"), newVec(2, 1, -1).U("k")),
			"ud",
			0, 0,
			"Derivative tower contracted with a base quadratic polynomial (2x-1)(x+2) =2x^2 + x -1 "},
		{E(DerivativeTower2.D("i").U("j").D("k"), newVec(2, 1, -1).U("k"), newVec(4, -2, 1).U("j")),
			"u",
			0, 0,
			"We can do arbitrary horizontal shifts by multiplying this matrix\n" +
				"by a shift vector <-disp^0, -disp^1 ...>. Here we shift to the right by 2."},
		{E(DerivativeTower2.D("i").U("j").D("k"), newVec(3, -18, -21).U("k")),
			"ud",
			0, 0,
			"Derivative tower contracted with a base quadratic polynomial (3x+3)(x-7) =3x^2-18x-21 "},
		{E(DerivativeTower2.D("i").U("j").D("k"), newVec(3, -18, -21).U("k"), newVec(25, 5, 1).U("j")),
			"u",
			0, 0,
			"We can do arbitrary horizontal shifts by multiplying this matrix\n" +
				"by a shift vector <-disp^0, -disp^1 ...>. Here we shift to the left by 5."},
	}
	for _, elt := range table {
		tensor, err := shmeh.Eval(elt.t...)
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

// 0th, first, and 2nd derivatvies on 2nd degree polynomials.
var DerivativeTower2 = shmeh.NewIntTensor(
	func(i ...int) int {
		z := [][][]int{
			{
				{0, 0, 0},
				{0, 0, 0},
				{2 / 2, 0, 0}}, // 1/n!
			{
				{0, 0, 0},
				{2, 0, 0},
				{0, 1, 0}},
			{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1}},
		}
		return z[i[0]][i[1]][i[2]]
	},
	"dud",
	[]int{3, 3, 3},
)

// New vector helper function.
func newVec(i ...int) *shmeh.Tensor {
	t := shmeh.NewIntTensor(
		func(j ...int) int {
			return i[j[0]]
		},
		"u",
		[]int{len(i)})
	return &t
}
