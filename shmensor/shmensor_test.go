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
	"reflect"
	"testing"
)

// New vector helper function.
func newVec(i ...int) Tensor {
	t := NewIntTensor(
		func(j ...int) int {
			return i[j[0]]
		},
		"u",
		[]int{len(i)})
	return t
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
		{Product(newVec(1, 2, 3), newRow(4, 5, 6)),
			0,
			1,
			[][]interface{}{{32}},
			"",
			[]int{},
			false},
	}
	for _, tt := range table {
		r, err := Trace(tt.tensor, tt.firstIndex, tt.secondIndex)
		// We're expecting an error.
		if tt.producesError && err == nil {
			t.Errorf("Trace attempt should have errored but did not.")
		}
		// First check the actual numerical values are correct.
		if !reflect.DeepEqual(r.reify(), tt.reified) {
			t.Errorf("Reified tensor value error: want %v, got %v", tt.reified, r.reify())
		}
		// Check signatures.
		if !reflect.DeepEqual(r.signature, tt.signature) {
			t.Errorf("Signature mismatch: want %v, got %v", tt.signature, r.signature)
		}
		// Check dimensions.
		if !reflect.DeepEqual(r.dim, tt.dimension) {
			t.Errorf("Dimension mismatch: want %v, got %v", tt.dimension, r.dim)
		}
	}
}
