/*
Copyright 2019 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"fmt"
	shmeh "shmensor/shmensor"
)

func main() {
	table := []struct {
		t    shmeh.Tensor
		desc string
	}{
		{col1, "Column Vector (1, 0)."},
		{row1, "Row Vector (0, 1)."},
		{mat1, "Matrix (1, 1)."},
		{bivec1, "Bivector (2, 0.)"},
		{shmeh.Eval(s1.U(""), x1.U("i").D("j"), x2.U("k").D("l")),
			"Evaluating a scalar times a tensor product of a row and column (2, 2)."},
	}

	for _, elt := range table {
		fmt.Printf("%v\n", elt.desc)
		fmt.Printf("%v", elt.t)
	}

	/*
		fmt.Printf("%v", mat1)
		fmt.Printf("%v", bivec1)
		fmt.Printf("%v", bilinearform1)
		fmt.Printf("%v", onetwo1)
		fmt.Printf("%v", x1)
		fmt.Printf("%v", x2)
		fmt.Printf("%v", shmeh.Product(x1, x2))

		x1 := shmeh.Eval(s1.U(""), x1.U(""), x2.D(""))
		fmt.Printf("%v", x1)*/
	/*
		fmt.Printf("%v", trace(product(x1, x2), 1, 2))
		scalar := trace(product(row1, col1), 0, 1)
		fmt.Printf("scalar %v \n", scalar)
		p := product(scalar, x2)
		fmt.Printf("%v", p)
		fmt.Printf("T %v", trace(p, 0, 1))*/
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

var col1 = shmeh.NewTensor(
	identity,
	"u",
	[]int{3},
)

var row1 = shmeh.NewTensor(
	identity,
	"d",
	[]int{3},
)

var mat1 = shmeh.NewTensor(
	identity,
	"ud",
	[]int{3, 3},
)

var bivec1 = shmeh.NewTensor(
	identity,
	"uu",
	[]int{3, 3},
)

var bilinearform1 = shmeh.NewTensor(
	identity,
	"dd",
	[]int{3, 3},
)

var onetwo1 = shmeh.NewTensor(
	identity,
	"udd",
	[]int{3, 3, 3},
)

var s1 = shmeh.NewTensor(
	func(i ...int) int {
		return 5
	},
	"",
	[]int{},
)

var x1 = shmeh.NewTensor(
	func(i ...int) int {
		z := [][]int{
			{1, 2},
			{3, 4}}
		return z[i[0]][i[1]]
	},
	"ud",
	[]int{2, 2},
)

var x2 = shmeh.NewTensor(
	func(i ...int) int {
		z := [][]int{
			{0, 5},
			{6, 7}}
		return z[i[0]][i[1]]
	},
	"ud",
	[]int{2, 2},
)
