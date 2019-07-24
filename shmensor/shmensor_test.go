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
package shmensor

import (
	"fmt"
	"math"
	"reflect"
	"testing"
)

/*
We define a bunch of Tensor helper functions before the tests.
*/

// New vector helper function.
func newVec(i ...int) *Tensor {
	t := NewIntTensor(
		func(j ...int) int {
			return i[j[0]]
		},
		"u",
		[]int{len(i)})
	return &t
}

// New row helper function.
func newRow(i ...int) Tensor {
	t := NewIntTensor(
		func(j ...int) int {
			return i[j[0]]
		},
		"d",
		[]int{len(i)})
	return t
}

// New scalar helper function.
func newScalar(i int) Tensor {
	t := NewIntTensor(
		func(j ...int) int {
			return i
		},
		"",
		[]int{})
	return t
}

// New matrix helper function.
func newMatrix(v [][]int) *Tensor {
	vals := make([][]int, len(v))
	copy(vals, v)
	t := NewIntTensor(
		func(i ...int) int {
			return vals[i[0]][i[1]]
		},
		"ud",
		[]int{len(v), len(v[0])},
	)
	return &t
}

// New string matrix helper function.
func newStringMatrix(v [][]string) *Tensor {
	vals := make([][]string, len(v))
	copy(vals, v)
	t := NewStringTensor(
		func(i ...int) string {
			return vals[i[0]][i[1]]
		},
		"ud",
		[]int{len(v), len(v[0])},
	)
	return &t
}

// New real matrix helper function.
func newRealMatrix(v [][]float64) *Tensor {
	vals := make([][]float64, len(v))
	copy(vals, v)
	t := NewRealTensor(
		func(i ...int) float64 {
			return vals[i[0]][i[1]]
		},
		"ud",
		[]int{len(v), len(v[0])},
	)
	return &t
}

var det1 = NewIntTensor(
	func(i ...int) int {
		if i[0] == i[1] {
			return i[0] + 1
		}
		return 0
	},
	"ud",
	[]int{3, 3},
)

// Levi-civita symbol on 3 letters.
var eps = NewIntTensor(
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

// Dirac Delta on 3 indices
func newIntDirac3(size int) *Tensor {
	f := func(i ...int) int {
		if i[0] == i[1] && i[1] == i[2] {
			return 1
		}
		return 0
	}

	t := NewIntTensor(
		f,
		"udd",
		[]int{size, size, size})
	return &t
}

// Test adding two tensors together.
func TestPlus(t *testing.T) {
	table := []struct {
		description string
		a           *Tensor
		b           *Tensor
		reified     [][]interface{}
		err         bool
	}{
		{
			"Adding two 2x2 string matrices.",
			newStringMatrix([][]string{
				{"a", "b"},
				{"c", "d"},
			}),
			newStringMatrix([][]string{
				{"w", "x"},
				{"y", "z"},
			}),
			[][]interface{}{
				{"a + w", "b + x"},
				{"c + y", "d + z"},
			},
			false,
		},
		{
			"Adding matrices of incompatible dimension.",
			newStringMatrix([][]string{
				{"a", "b", "c"},
				{"d", "e", "f"},
			}),
			newStringMatrix([][]string{
				{"w", "x"},
				{"y", "z"},
			}),
			[][]interface{}{},
			true,
		},
		{
			"Adding matrices of incompatible type.",
			newStringMatrix([][]string{
				{"a", "b"},
				{"d", "e"},
			}),
			newMatrix([][]int{
				{1, 2},
				{3, 4},
			}),
			[][]interface{}{},
			true,
		},
	}
	for _, tt := range table {
		p, err := Plus(*tt.a, *tt.b)
		if err != nil && !tt.err {
			t.Errorf("In %v, got err%v.\n Got an error in the plus test when not expecting one.",
				tt.description, err)
		}
		if err == nil && tt.err {
			t.Errorf("On %v | Expected an error in the plus test but didn't get one.",
				tt.description)
		}
		// If was expecting an error and caught one, keep going.
		if tt.err {
			continue
		}
		s := p.Reify()
		if !reflect.DeepEqual(s, tt.reified) {
			t.Errorf("On %v: got %v, want %v", tt.description, s, tt.reified)
		}
	}
}

// There are three types of trace errors that can happen.
// a) One or both of the indices you want to contract don't exist.
// b) The indices are of different dimension.
// c) The indices aren't of opposite variance (ud or du).
func TestTrace(t *testing.T) {
	table := []struct {
		tensor        Tensor
		firstIndex    int
		secondIndex   int
		reified       [][]interface{}
		signature     string
		dimension     []int
		producesError bool
	}{
		{Product(*newVec(1, 2, 3), newRow(4, 5, 6), &Profiler{}),
			0,
			1,
			[][]interface{}{{32}},
			"",
			[]int{},
			false},
	}
	for _, tt := range table {
		r, err := Trace(tt.tensor, tt.firstIndex, tt.secondIndex, &Profiler{})
		// We're expecting an error.
		if tt.producesError && err == nil {
			t.Errorf("Trace attempt should have errored but did not.")
		}
		// First check the actual numerical values are correct.
		if !reflect.DeepEqual(r.Reify(), tt.reified) {
			t.Errorf("Reified tensor value error: want %v, got %v", tt.reified, r.Reify())
		}
		// Check signatures.
		if !reflect.DeepEqual(r.Signature(), tt.signature) {
			t.Errorf("Signature mismatch: want %v, got %v", tt.signature, r.Signature())
		}
		// Check dimensions.
		if !reflect.DeepEqual(r.Dimension(), tt.dimension) {
			t.Errorf("Dimension mismatch: want %v, got %v", tt.dimension, r.Dimension())
		}
	}
}

// Test applying a function to every entry of a tensor.
// Useful for stuff like sigmoid functions in neural nets.
func TestApply(t *testing.T) {
	table := []struct {
		description string
		f           Function
		t           *Tensor
		reified     [][]interface{}
		err         bool
	}{
		{"Add bars to entries.",
			NewStringFunction(func(s string) string {
				return "|" + s + "|"
			}),
			newStringMatrix([][]string{
				{"a", "b"},
				{"c", "d"},
			}),
			[][]interface{}{
				{"|a|", "|b|"},
				{"|c|", "|d|"},
			},
			false,
		},
		{"Square every entry of a real tensor",
			NewRealFunction(func(r float64) float64 {
				return r * r
			}),
			newRealMatrix([][]float64{
				{2., -1., 0.},
				{6., 8., 2.},
			}),
			[][]interface{}{
				{4., 1., 0.},
				{36., 64., 4.},
			},
			false,
		},
	}

	for _, tt := range table {
		r, err := Apply(tt.f, *tt.t)
		if err != nil && !tt.err {
			t.Errorf("In %v, got err%v.\n Got an error in the TestApply when not expecting one.",
				tt.description, err)
		}
		if err == nil && tt.err {
			t.Errorf("On %v | Expected an error in TestApply but didn't get one.",
				tt.description)
		}
		// If it was expecting an error and caught one, keep going.
		if tt.err {
			continue
		}
		real := r.Reify()
		if !reflect.DeepEqual(real, tt.reified) {
			t.Errorf("On %v: got %v, want %v", tt.description, real, tt.reified)
		}
	}
}

// Reify has three properites:
//
func TestReify(t *testing.T) {
	// Given signature and dimensions, create a dummy tensor
	testTensor := func(signature string, dimensions []int) Tensor {
		f := func(i ...int) int {
			// Prepend the digits with 9
			value := 9 * int(math.Pow10(len(i)))
			for place, j := range i {
				value += j * int(math.Pow10(len(i)-place-1))
			}
			return value
		}
		t := NewIntTensor(
			f,
			signature,
			dimensions)
		return t
	}

	table := []struct {
		description string
		tensor      Tensor
		reified     [][]interface{}
	}{
		{
			"Int vector",
			*newVec(2, 3, 4, 5, 6),
			[][]interface{}{
				{2}, {3}, {4}, {5}, {6},
			},
		},
		{
			"Int vector",
			Product(*newVec(1, 2, 3), *newVec(1, 2, 3), nil),
			[][]interface{}{
				{1},
				{2},
				{3},
				{2},
				{4},
				{6},
				{3},
				{6},
				{9},
			},
		},
		{
			"Int row",
			newRow(1, 3, 5, 7, 9),
			[][]interface{}{
				{1, 3, 5, 7, 9},
			},
		},
		{
			"Test Tensor 1",
			testTensor("dud", []int{4, 2, 2}),
			[][]interface{}{
				{9000, 9001, 9100, 9101, 9200, 9201, 9300, 9301},
				{9010, 9011, 9110, 9111, 9210, 9211, 9310, 9311},
			},
		},
		{
			"Test Tensor 2",
			testTensor("ddu", []int{4, 2, 2}),
			[][]interface{}{
				{9000, 9010, 9100, 9110, 9200, 9210, 9300, 9310},
				{9001, 9011, 9101, 9111, 9201, 9211, 9301, 9311},
			},
		},
		{
			"Test Tensor 3",
			testTensor("uuudd", []int{2, 2, 2, 3, 2}),
			[][]interface{}{
				{900000, 900001, 900010, 900011, 900020, 900021},
				{900100, 900101, 900110, 900111, 900120, 900121},
				{901000, 901001, 901010, 901011, 901020, 901021},
				{901100, 901101, 901110, 901111, 901120, 901121},
				{910000, 910001, 910010, 910011, 910020, 910021},
				{910100, 910101, 910110, 910111, 910120, 910121},
				{911000, 911001, 911010, 911011, 911020, 911021},
				{911100, 911101, 911110, 911111, 911120, 911121},
			},
		},
	}

	for _, tt := range table {
		r := tt.tensor.Reify()
		if !reflect.DeepEqual(r, tt.reified) {
			// Pretty print
			got := "\n"
			for _, row := range r {
				got += fmt.Sprintf("\n")
				for _, elt := range row {
					got += fmt.Sprintf("%v, ", elt)
				}
			}
			want := "\n"
			for _, row := range tt.reified {
				want += fmt.Sprintf("\n")
				for _, elt := range row {
					want += fmt.Sprintf("%v, ", elt)
				}
			}

			t.Errorf("On %v\t%v\t%v\n got %v, want %v", tt.description,
				tt.tensor.Signature(), tt.tensor.Dimension(),
				got, want)

		}
	}
}

// Test evaluating tensors in abstract index notation.
// Consider this a sort of integration test, as it exercises most public functions.
func TestEval(t *testing.T) {
	table := []struct {
		description      string
		tensorExpression Term
		reified          [][]interface{}
		signature        string
		dimension        []int
		producesError    bool
	}{
		{
			"Determinant on [1, 2] [3,4]. Should be 6 but we don't scale so 36.",
			E(eps.D("ijk"), eps.D("pqr"),
				det1.U("p").D("i"),
				det1.U("q").D("j"),
				det1.U("r").D("k")),
			[][]interface{}{{36}},
			"",
			[]int{},
			false},

		{
			"Cross product of <2,3,4> and <5,6,7> in abstract index notation.",
			E(eps.D("ijk"),
				newVec(2, 3, 4).U("j"),
				newVec(5, 6, 7).U("k")),
			[][]interface{}{{-3, 6, -3}},
			"d",
			[]int{3},
			false},

		{
			"Component-wise multiplication (Hadamard product)",
			E(newIntDirac3(5).U("a").D("b").D("c"),
				newVec(1, 2, 3, 4, 5).U("b"),
				newVec(1, 2, 3, 4, 5).U("c")),
			[][]interface{}{{1}, {4}, {9}, {16}, {25}},
			"u",
			[]int{5},
			false},

		/*
			E(dirac_delta3.U("a").D("b").D("c"),
						newVec(1, 2, 3, 4, 5).U("b"),
						newVec(1, 2, 3, 4, 5).U("c")),
						"Hadamard Product (component wise multiplication) on <1,2,3,4,5>."},
		*/
	}
	for _, tt := range table {
		tensor, err, _ := tt.tensorExpression.Eval()
		// We're expecting an error.
		if tt.producesError && err == nil {
			t.Errorf("On %v\n Trace attempt should have errored but did not.", tt.description)
		}
		// First check the actual numerical values are correct.
		if !reflect.DeepEqual(tensor.Reify(), tt.reified) {
			t.Errorf("On %v\n Reified tensor value error: want %v, got %v", tt.description, tt.reified, tensor.Reify())
		}
		// Check signatures.
		if !reflect.DeepEqual(tensor.Signature(), tt.signature) {
			t.Errorf("On %v\n Signature mismatch: want %v, got %v", tt.description, tt.signature, tensor.Signature())
		}
		// Check dimensions.
		if !reflect.DeepEqual(tensor.Dimension(), tt.dimension) {
			t.Errorf("On %v\n Dimension mismatch: want %v, got %v", tt.description, tt.dimension, tensor.Dimension())
		}
	}
}
