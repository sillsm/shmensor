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
	"reflect"
	shmeh "shmensor/shmensor"
)

func E(e ...shmeh.Expression) []shmeh.Expression {
	return e
}

func main() {
	table := []struct {
		t    []shmeh.Expression
		desc string
	}{
		{E(col1.U("i")), "Column Vector (1, 0)."},
		{E(row1.D("j")), "Row Vector (0, 1)."},
		{E(mat1.U("i").D("j")), "Matrix (1, 1)."},
		{E(bivec1.U("i").U("j")), "Bivector (2, 0.)"},
		{E(s1.U(""), x1.U("i").D("j"), x2.U("k").D("l")),
			"Evaluating a scalar times a tensor product of a row and column (2, 2)."},
		// equals 36 until you divide by 6. To be supported later when extended to floats and bignum
		{E(dirac_delta3.U("a").D("b").D("c"),
			newVec(1, 2, 3, 4, 5).U("b"),
			newVec(1, 2, 3, 4, 5).U("c")),
			"Hadamard Product (component wise multiplication) on <1,2,3,4,5>."},
		//
		{E(eps.D("ijk"), eps.D("pqr"),
			det1.U("p").D("i"),
			det1.U("q").D("j"),
			det1.U("r").D("k")),
			"Determinant of <1,2><3,4> in abstract index notation."},
		//https://www.mathsisfun.com/algebra/vectors-cross-product.html
		{E(eps.D("ijk"),
			newVec(2, 3, 4).U("j"),
			newVec(5, 6, 7).U("k")),
			"Cross product of <2,3,4> and <5,6,7> in abstract index notation."},
		//https://www.wolframalpha.com/input/?i=%7B%7B2+%2B+6.5i,+1+-7.001i%7D,+%7B0,+3+%2B+i%7D%7D+*+%7B%7Bi%7D,%7B6%7D%7D
		{E(complex1.U("i").D("j")), "Complex LHS"},
		{E(complex2.D("j")), "Complex RHS"},
		{E(complex1.U("i").D("j"), complex2.D("j")),
			"Their product."},
	}

	for _, elt := range table {
		tensor, err := shmeh.Eval(elt.t...)
		if err != nil {
			panic(err)
		}
		fmt.Printf("%v\n", elt.desc)
		fmt.Printf("%v", tensor)
	}
}

func identity(i ...int) int {
	val := i[0]
	for _, elt := range i {
		if elt != val {
			return 0
		}
	}
	return 1
}

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

// Levi-civita symbol on 3 letters.
var eps = shmeh.NewIntTensor(
	func(i ...int) int {
		if reflect.DeepEqual(i, []int{0, 1, 2}) {
			return 1
		}
		if reflect.DeepEqual(i, []int{1, 2, 0}) {
			return 1
		}
		if reflect.DeepEqual(i, []int{2, 0, 1}) {
			return 1
		}
		if reflect.DeepEqual(i, []int{2, 1, 0}) {
			return -1
		}
		if reflect.DeepEqual(i, []int{1, 0, 2}) {
			return -1
		}
		if reflect.DeepEqual(i, []int{0, 2, 1}) {
			return -1
		}
		return 0
	},
	"ddd",
	[]int{3, 3, 3})

var col1 = shmeh.NewIntTensor(
	identity,
	"u",
	[]int{3},
)

var row1 = shmeh.NewIntTensor(
	identity,
	"d",
	[]int{3},
)

var mat1 = shmeh.NewIntTensor(
	identity,
	"ud",
	[]int{3, 3},
)

var dirac_delta3 = shmeh.NewIntTensor(
	identity,
	"udd",
	[]int{5, 5, 5},
)

var bivec1 = shmeh.NewIntTensor(
	identity,
	"uu",
	[]int{3, 3},
)

var bilinearform1 = shmeh.NewIntTensor(
	identity,
	"dd",
	[]int{3, 3},
)

var onetwo1 = shmeh.NewIntTensor(
	identity,
	"udd",
	[]int{3, 3, 3},
)

var s1 = shmeh.NewIntTensor(
	func(i ...int) int {
		return 5
	},
	"",
	[]int{},
)

var x1 = shmeh.NewIntTensor(
	func(i ...int) int {
		z := [][]int{
			{1, 2},
			{3, 4}}
		return z[i[0]][i[1]]
	},
	"ud",
	[]int{2, 2},
)

var x2 = shmeh.NewIntTensor(
	func(i ...int) int {
		z := [][]int{
			{0, 5},
			{6, 7}}
		return z[i[0]][i[1]]
	},
	"ud",
	[]int{2, 2},
)

var det1 = shmeh.NewIntTensor(
	func(i ...int) int {
		if i[0] == i[1] {
			return i[0] + 1
		}
		return 0
	},
	"ud",
	[]int{3, 3},
)

var complex1 = shmeh.NewComplexTensor(
	func(i ...int) complex128 {
		z := [][]complex128{
			{complex(2, 6.5), complex(1, -7.001)},
			{complex(0, 0), complex(3, 1)}}
		return z[i[0]][i[1]]
	},
	"ud",
	[]int{2, 2},
)

var complex2 = shmeh.NewComplexTensor(
	func(i ...int) complex128 {
		z := []complex128{complex(0, 1), complex(6, 0)}
		return z[i[0]]
	},
	"u",
	[]int{2},
)
