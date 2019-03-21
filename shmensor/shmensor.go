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
package shmensor

import (
	"fmt"
	"log"
	"sort"
	"strings"
)

// A Tensor is not a matrix. It's a tensor.
// A Tensor is a multi-dimensional array that
// transforms according to the co and contravariant.
// tensor transformation laws.
type Tensor struct {
	//coordinate function
	f func(i ...int) int
	// signature like "uddduu"
	// combo of contra and co indices
	signature string
	// dimension of indices.
	// len(dim)==len(signature)
	dim []int
}

// An Expression is not a Tensor. It's an Expression symbolizing a
// desired combination of tensor products and contractions on Tensors.
// When you evaluate it, you get a Tensor.
// Abstract index notation with Einstein summation.
type Expression struct {
	t         *Tensor
	indices   string
	signature string
}

func NewTensor(f func(i ...int) int, signature string, dim []int) Tensor {
	return Tensor{
		f,
		signature,
		dim,
	}
}

// Pretty printing.
func (t Tensor) String() string {
	ret := "\n"
	fmt.Printf("%v\n", t.reify())
	ret += fmt.Sprintf("%v signature \n\n", t.signature)
	return ret
}

// Like t.U("ij").D("k").U("a").D("b")
// Like t1.U("jk").U("arb").D("xyz")
// Eval(t, t1)
// Eval(t.U("ij").D("k").U("a").D("b"), t1.U("jk").U("arb").D("xyz"))
// Mat multiply example
// Eval(t1.U("i"), t2.D("j"))
// Transpose like
// Eval(t1.I().U("j").D("i"))
func (t *Tensor) U(indices string) Expression {
	return Expression{t, "", ""}.U(indices)
}

func (t *Tensor) D(indices string) Expression {
	return Expression{t, "", ""}.D(indices)
}

func (e Expression) U(indices string) Expression {
	for range indices {
		e.signature += "u"
	}
	e.indices += indices
	return e
}

func (e Expression) D(indices string) Expression {
	for range indices {
		e.signature += "d"
	}
	e.indices += indices
	return e
}

// Eval takes a list of expressions
// representing a solo or product term
// of tensors in abstract index notation
// and returns the resulting Tensor.
//
// Note that shmensor Tensors are lazy, so computation
// is only performed when you call Reify().
//
// Consider verbose mode boolean to explore what's happening.
func Eval(t ...Expression) Tensor {
	// Eval first tensors products
	if len(t) == 0 {
		return Tensor{}
	}
	// TODO(Max): Eventually sequences of products could use dynamic programming.
	head, tail := *t[0].t, t[1:]
	for _, elt := range tail {
		head = Product(head, *elt.t)
	}
	// Now your left with a single big product tensor head.
	// Next, figure out repeated indices on head and keep contracting until you can't.
	// Make sure to fail if any of the conditions don't hold:
	// 1. Any letter only allowed to be repeated twice. ijjk ok, ijjj not ok.
	// 2. Repeated indices must have opposite orientation (up or down)
	trueSignature := ""
	signature := ""
	indices := ""
	for _, elt := range t {
		trueSignature += elt.t.signature
		signature += elt.signature
		indices += elt.indices
	}

	sortedIndices := strings.Split(indices, "")
	sort.Strings(sortedIndices)
	//signatures := strings.Split(signature, "")

	seen := make(map[string]int)
	seen = seen

	fmt.Printf("True Signature: %v\n", trueSignature)
	fmt.Printf("Desired Signature: %v\n", signature)
	fmt.Printf("Indices: %v\n", indices)

	/*
		strings.Split(s, "")
		fmt.Printf("%v\n", x)
		sort.Strings(x)
		fmt.Printf("%v\n", x)
	*/

	return head
}

// Trace is a contraction on two indices.
// eventually should forbid callers
// from accessing directly
// and/or verify indices exist to contract
// and are same dimensions.
func Trace(t Tensor, a, b int) Tensor {
	// assume a less than b
	if b < a {
		b, a = a, b
	}
	if t.dim[a] != t.dim[b] {
		log.Fatalf("trace error incompatible dims")
	}

	g := func(i ...int) int { //takes in dim 2 less
		sum := 0
		// Shift args for the 2 repeated indices.
		inner := make([]int, len(i))
		copy(inner, i)
		inner = append(inner, 0, 0)
		copy(inner[a+1:], inner[a:])
		copy(inner[b+1:], inner[b:])

		for k := 0; k < t.dim[a]; k++ {
			inner[a], inner[b] = k, k
			sum += t.f(inner...)
		}
		return sum
	}

	var sig string
	for i, elt := range t.signature {
		if i != a && i != b {
			sig += string(elt)
		}
	}

	// delete a and b from the dim array
	d := make([]int, len(t.dim))
	copy(d, t.dim)
	d = append(d[:b], d[b+1:]...)
	d = append(d[:a], d[a+1:]...)

	return Tensor{
		g,
		sig,
		d,
	}
}

func Product(t1, t2 Tensor) Tensor {
	f := func(inner ...int) int {
		i := make([]int, len(inner))
		copy(i, inner)
		return t1.f(i[0:len(t1.dim)]...) *
			t2.f(i[len(t1.dim):]...)
	}

	t2dim := make([]int, len(t2.dim))
	copy(t2dim, t2.dim)
	return Tensor{
		f,
		t1.signature + t2.signature,
		append(t1.dim, t2dim...),
	}
}

//
//func Eval(t1, t2 Tensor) Tensor {

// given number and dimensions, return co or contravariant
// coordinate. Up to caller to merge one with
// the other.
func giveCoordinate(dim []int, i int) []int {
	//fmt.Printf("GC %v\t%v\n",dim,i)
	val := 1
	newDim := make([]int, len(dim))
	for j := len(dim) - 1; j >= 0; j-- {
		val, newDim[j] = dim[j]*val, val
	}

	c := i
	var coord []int
	for _, d := range newDim {
		coord = append(coord, c/d)
		c = c - (c/d)*d
	}
	return coord
}

// This method is the one most badly in need of testing.
// "Reify" is one of *many* possible ways to visualize a tensor as a matrix.
// The horizontal and vertical indices correspond to walking the covariant
// and contravariant indices of the tensor.
//
// E.g. If a tensor has 3 contravariant indices of dimension 1 | 2 \ 3
// and 1 covariant index of dimension 4, Reify will produce a
// 4 by (1*2*3)=6 matrix, with the indices sorted.
// Do not treat this as a real matrix it's merely for convenience.
func (t Tensor) reify() [][]int {
	coDim := 1
	contraDim := 1
	var co []int
	var contra []int
	for i, elt := range t.signature {
		if string(elt) == "u" {
			contraDim *= t.dim[i]
			contra = append(contra, t.dim[i])
		}
		if string(elt) == "d" {
			coDim *= t.dim[i]
			co = append(co, t.dim[i])
		}
	}
	// Make kroeneker representation
	twoD := make([][]int, contraDim)
	for i := 0; i < contraDim; i++ {
		twoD[i] = make([]int, coDim)
	}

	for i := 0; i < contraDim; i++ {
		for j := 0; j < coDim; j++ {
			coStack := giveCoordinate(co, j)
			contraStack := giveCoordinate(contra, i)

			// now merge
			var merged []int
			for _, elt := range t.signature {
				if string(elt) == "u" {
					merged = append(merged, contraStack[0])
					contraStack = contraStack[1:]
				}
				if string(elt) == "d" {
					merged = append(merged, coStack[0])
					coStack = coStack[1:]
				}
			}
			twoD[i][j] = t.f(merged...)
		}
	}
	return twoD
}
