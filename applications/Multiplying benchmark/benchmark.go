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
		{E(twobytwo.U("a").D("b"), twobytwo.U("b").D("c"), twobytwo.U("c").D("d"),
			twobytwo.U("d").D("e"), twobytwo.U("e").D("f"), twobytwo.U("f").D("g"),
			twobytwo.U("g").D("h"), twobytwo.U("h").D("i"), twobytwo.U("i").D("j"),
			twobytwo.U("j").D("k"), twobytwo.U("k").D("l"), twobytwo.U("l").D("m"),
			twobytwo.U("m").D("n"), twobytwo.U("n").D("o"), twobytwo.U("o").D("p"),
			twobytwo.U("p").D("q"), twobytwo.U("q").D("r"), twobytwo.U("r").D("s"),
			twobytwo.U("s").D("t"), twobytwo.U("t").D("u"), twobytwo.U("u").D("v"),
		),
			"ud",
			0, 0,
			"Multiplying the same 2 x 2 matrix 21 times."},
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

var twobytwo = shmeh.NewIntTensor(
	func(i ...int) int {
		z := [][]int{
			{1, 2},
			{3, 4},
		}
		return z[i[0]][i[1]]
	},
	"ud",
	[]int{2, 2},
)

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
