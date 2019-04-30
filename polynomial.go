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

func main() {
	fmt.Printf("%v", shmeh.Eval(D3.D("ij"), P1.U("j")))
	fmt.Printf("%v", shmeh.Eval(D3.U("i").D("k"), P2.U("m").D("k")))
	fmt.Printf("%v", shmeh.Eval(D3.U("i").D("j"), P3.U("k").D("j").D("m")))

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
				{0, 1, 0},  // x
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
	"udu",
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
