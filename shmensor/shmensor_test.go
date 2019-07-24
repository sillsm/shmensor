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
		plus        Plus
		reified     [][]interface{}
		err         bool
	}{
		{
			"Adding two 2x2 string matrices.",
			Plus{
				newStringMatrix([][]string{
					{"a", "b"},
					{"c", "d"},
				}),
				newStringMatrix([][]string{
					{"w", "x"},
					{"y", "z"},
				})},
			[][]interface{}{
				{"a + w", "b + x"},
				{"c + y", "d + z"},
			},
			false,
		},
		{
			"Adding matrices of incompatible dimension.",
			Plus{
				newStringMatrix([][]string{
					{"a", "b", "c"},
					{"d", "e", "f"},
				}),
				newStringMatrix([][]string{
					{"w", "x"},
					{"y", "z"},
				})},
			[][]interface{}{},
			true,
		},
		{
			"Adding matrices of incompatible type.",
			Plus{
				newStringMatrix([][]string{
					{"a", "b"},
					{"d", "e"},
				}),
				newMatrix([][]int{
					{1, 2},
					{3, 4},
				})},
			[][]interface{}{},
			true,
		},
	}
	for _, tt := range table {
		p, err, _ := tt.plus.Eval()
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
		apply       Apply
		reified     [][]interface{}
		err         bool
	}{
		{"Add bars to entries.",
			Apply{
				NewStringFunction(func(s string) string {
					return "|" + s + "|"
				}),
				newStringMatrix([][]string{
					{"a", "b"},
					{"c", "d"},
				})},
			[][]interface{}{
				{"|a|", "|b|"},
				{"|c|", "|d|"},
			},
			false,
		},
		{"Square every entry of a real tensor",
			Apply{
				NewRealFunction(func(r float64) float64 {
					return r * r
				}),
				newRealMatrix([][]float64{
					{2., -1., 0.},
					{6., 8., 2.},
				})},
			[][]interface{}{
				{4., 1., 0.},
				{36., 64., 4.},
			},
			false,
		},
	}

	for _, tt := range table {
		r, err, _ := tt.apply.Eval()
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
	// where each value is the string form of its index.
	testTensor := func(signature string, dimensions []int) Tensor {
		f := func(i ...int) string {
			value := ""
			for _, j := range i {
				value += fmt.Sprintf("%v", j)
			}
			return value
		}
		t := NewStringTensor(
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
				{"000", "001", "100", "101", "200", "201", "300", "301"},
				{"010", "011", "110", "111", "210", "211", "310", "311"},
			},
		},
		{
			"Test Tensor 2",
			testTensor("ddu", []int{4, 2, 2}),
			[][]interface{}{
				{"000", "010", "100", "110", "200", "210", "300", "310"},
				{"001", "011", "101", "111", "201", "211", "301", "311"},
			},
		},
		{
			"Test Tensor 3",
			testTensor("uuudd", []int{2, 2, 2, 3, 2}),
			[][]interface{}{
				{"00000", "00001", "00010", "00011", "00020", "00021"},
				{"00100", "00101", "00110", "00111", "00120", "00121"},
				{"01000", "01001", "01010", "01011", "01020", "01021"},
				{"01100", "01101", "01110", "01111", "01120", "01121"},
				{"10000", "10001", "10010", "10011", "10020", "10021"},
				{"10100", "10101", "10110", "10111", "10120", "10121"},
				{"11000", "11001", "11010", "11011", "11020", "11021"},
				{"11100", "11101", "11110", "11111", "11120", "11121"},
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
