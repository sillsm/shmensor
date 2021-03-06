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
	"index/suffixarray"
	"log"
	"reflect"
	"sort"
)

// Type defines a ring element. To implement the interface
// define multiplication and addition of the ring elements.
//
// Several default Tensor types and initializer functions
// are defined at the end of the file.
//
// Note that most applications expect the inputs and output
// to be of the same type. You have to enforce that at
// object construction.
type Type interface {
	// Take two things, like numbers, get a new thing.
	Multiply(interface{}, interface{}) interface{}
	// Take two things, like numbers, get a new thing.
	Add(interface{}, interface{}) interface{}
}

// A Tensor is not a matrix. It's a tensor.
// A Tensor is a multi-dimensional array that
// transforms according to the co and contravariant.
// tensor transformation laws.
type Tensor struct {
	//coordinate function
	f func(i ...int) interface{}
	// signature like "uddduu"
	// combo of contra and co indices
	signature string
	// dimension of indices.
	// len(dim)==len(signature)
	dim []int
	// Type of Tensor elt. Like real number, complex number
	// rational number.
	t Type
}
type Evaluator interface {
	Eval() (Tensor, error, *Profiler)
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

// A Term is a list of Expressions. It represents a single term
// of tensor products and contractions in abstract index notation.
type Term struct {
	List []Expression
}

// Plus contains the sum of two Terms.
type Plus struct {
	A, B Evaluator
}

// Apply contains a function and an
// Evaluator interface to apply it to.
type Apply struct {
	Func Function
	E    Evaluator
}

// Function is a function that can be applied to every element of a Tensor.
// As of now, a caller can't make their own. They have to use a
// constructor from the types file which ensures types are aligned.
type Function struct {
	f func(interface{}) interface{}
	t Type
}

// Profiler collects metrics about Tensor evaluation for
// debugging and be
type Profiler struct {
	Multiplies int
	Adds       int
	Mutex      bool
	TraceCache int
}

// Pretty printing.
func (p *Profiler) String() string {
	s := fmt.Sprintf("Muls: %v\t Adds: %v\t Cached Traces Hits: %v", p.Multiplies, p.Adds, p.TraceCache)
	return s
}

// Pretty printing.
func (t Tensor) String() string {
	ret := "\n"
	grid := t.Reify()
	for _, row := range grid {
		ret += fmt.Sprintf("%v\n", row)
	}
	//fmt.Printf("%v", t.Reify())
	ret += fmt.Sprintf("Signature: \"%v\"\n\n", t.signature)
	return ret
}

// Some getters.
func (t Tensor) Signature() string {
	return t.signature
}

func (t Tensor) Dimension() []int {
	return t.dim
}

func (t Tensor) ContravariantIndices() []int {
	var ret []int
	for i, ch := range t.signature {
		if string(ch) == "u" {
			ret = append(ret, t.dim[i])
		}
	}
	return ret
}

func (t Tensor) CovariantIndices() []int {
	var ret []int
	for i, ch := range t.signature {
		if string(ch) == "d" {
			ret = append(ret, t.dim[i])
		}
	}
	return ret
}

func (t *Tensor) Reshape(signature string) {
	if len(signature) != len(t.signature) {
		fmt.Printf("Woops Tensor has sig %v\n", t.signature)
		panic("Trying to perform invalid reshape.")
	}
	t.signature = signature
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

// E wraps up a bunch of expressions into a term.
func E(i ...Expression) Term {
	return Term{i}
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
func (term Term) Eval() (Tensor, error, *Profiler) {
	// Profiler
	profiler := &Profiler{}
	t := term.List

	// Eval first tensors products
	if len(t) == 0 {
		return Tensor{}, nil, nil
	}

	/*
		Evaluation subroutines
	*/

	// Products two tensors together, returns the result.
	productSubroutine := func(e1, e2 *Expression) *Expression {
		t := Product(*e1.t, *e2.t, profiler)
		return &Expression{
			&t,
			e1.indices + e2.indices,
			e1.signature + e2.signature,
		}
	}

	// Contract the first contraction you can in the expression.
	// Return false if there was nothing to contract.
	traceSubroutine := func(e *Expression) (bool, error) {
		index := suffixarray.New([]byte(e.indices))
		for _, ch := range e.indices {
			offsets := index.Lookup([]byte(string(ch)), -1)
			sort.Ints(offsets)
			if len(offsets) > 2 {
				return false, fmt.Errorf("%v, Index repeated more than twice", string(ch))
			}
			if len(offsets) == 2 {
				// Do the contraction logic.
				traced, err := Trace(*e.t, offsets[0], offsets[1], profiler)
				e.t = &traced
				e.signature = e.signature[0:offsets[1]] + e.signature[offsets[1]+1:]
				e.signature = e.signature[0:offsets[0]] + e.signature[offsets[0]+1:]
				e.indices = e.indices[0:offsets[1]] + e.indices[offsets[1]+1:]
				e.indices = e.indices[0:offsets[0]] + e.indices[offsets[0]+1:]
				if err != nil {
					return true, err
				}
				return true, nil
			}
		}
		return false, nil
	}

	/*
		Evaluation strategy
	*/
	// Product everything together, then contract repeated indices until you can't.
	rhs := &Expression{t[len(t)-1].t, t[len(t)-1].indices, t[len(t)-1].signature}
	for i := len(t) - 1; i >= 0; i-- {
		if i == len(t)-1 {
			continue
		}
		rhs = productSubroutine(&t[i], rhs)
		//fmt.Printf("RHS pre: %v \n", rhs.indices)
		for {
			ok, err := traceSubroutine(rhs)
			if err != nil {
				panic(err)
			}
			if !ok {
				break
			}
		}
		//fmt.Printf("RHS post: %v \n", rhs.indices)

	}
	return *rhs.t, nil, profiler
}

// More pedestrian eval functions
func (e Expression) Eval() (Tensor, error, *Profiler) {
	return *e.t, nil, nil
}

func (t Tensor) Eval() (Tensor, error, *Profiler) {
	return t, nil, nil
}

func (as Apply) Eval() (Tensor, error, *Profiler) {
	apply := func(function Function, t Tensor) (Tensor, error) {
		if !reflect.DeepEqual(t.t, function.t) {
			return Tensor{}, fmt.Errorf("Tried to apply a function to a tensor"+
				" of incompatible types. %v %v",
				reflect.TypeOf(function.t), reflect.TypeOf(t.t))
		}

		f := func(inner ...int) interface{} {
			i := make([]int, len(inner))
			copy(i, inner)
			// Assert they are the same type here
			// TODO(xam)
			return function.f(t.f(i...))
		}
		return Tensor{
			f,
			t.signature,
			t.dim,
			t.t,
		}, nil
	}

	t, e1, p := as.E.Eval()
	t2, e2 := apply(as.Func, t)
	if e1 != nil || e2 != nil {
		e2 = fmt.Errorf("One of 2 possible errors stemming from function application"+
			"1)Evaling argument or 2)function application 1)%v, 2)%v, 3)%v", e1, e2)
	}
	return t2, e2, p

}

func (ps Plus) Eval() (Tensor, error, *Profiler) {
	plus := func(t1, t2 Tensor) (Tensor, error) {
		if !reflect.DeepEqual(t1.dim, t2.dim) {
			return Tensor{}, fmt.Errorf("Tried to add tensors of incompatible dimension. %v %v", t1, t2)
		}
		if !reflect.DeepEqual(t1.signature, t2.signature) {
			return Tensor{}, fmt.Errorf("Tried to add tensors of incompatible signature. %v %v", t1, t2)
		}
		if !reflect.DeepEqual(t1.t, t2.t) {
			return Tensor{}, fmt.Errorf("Tried to add tensors of incompatible type. %v %v",
				reflect.TypeOf(t1.t), reflect.TypeOf(t2.t))
		}
		f := func(inner ...int) interface{} {
			i := make([]int, len(inner))
			copy(i, inner)
			// Assert they are the same type here
			// TODO(xam)
			return t1.t.Add(t1.f(i...), t2.f(i...))
		}
		return Tensor{
			f,
			t1.signature,
			t1.dim,
			t1.t,
		}, nil
	}

	t1, e1, p1 := ps.A.Eval()
	t2, e2, p2 := ps.B.Eval()
	p3 := &Profiler{}
	if p1 != nil && p2 == nil {
		p3 = p1
	}
	if p1 == nil && p2 != nil {
		p3 = p2
	}
	if p1 != nil && p2 != nil {
		p3 = &Profiler{
			p1.Multiplies + p2.Multiplies,
			p1.Adds + p2.Adds,
			false,
			p1.TraceCache + p2.TraceCache,
		}
	}
	t3, e3 := plus(t1, t2)
	if e1 != nil || e2 != nil || e3 != nil {
		e3 = fmt.Errorf("One of 3 possible errors stemming from evaluating"+
			"1)first term, 2)second term, or their 3)sum.\n 1)%v, 2)%v, 3)%v", e1, e2, e3)
	}
	return t3, e3, p3
}

// Transpose swaps two tensor indices.
func Transpose(t Tensor, a, b int) (Tensor, error) {
	// assume a less than b
	if b < a {
		b, a = a, b
	}

	if b > len(t.signature)-1 || a < 0 {
		panic("Trying to transpose bad indices.")
	}
	g := func(i ...int) interface{} { //takes in dim 2 less
		inner := make([]int, len(i))
		copy(inner, i)
		inner[a], inner[b] = inner[b], inner[a]
		return t.f(inner...)
	}

	// Must swap dim
	t.dim[a], t.dim[b] = t.dim[b], t.dim[a]

	return Tensor{
		g,
		t.signature,
		t.dim,
		t.t,
	}, nil
}

// Trace is a contraction on two indices.
// eventually should forbid callers
// from accessing directly
// and/or verify indices exist to contract
// and are same dimensions.
func Trace(t Tensor, a, b int, profiler *Profiler) (Tensor, error) {

	// assume a less than b
	if b < a {
		b, a = a, b
	}
	if t.dim[a] != t.dim[b] {
		log.Fatalf("trace error incompatible dims %v, %v", t.dim[a], t.dim[b])
	}

	cache := make(map[string]interface{})
	g := func(i ...int) interface{} { //takes in dim 2 less
		var sum interface{}
		// Shift args for the 2 repeated indices.
		inner := make([]int, len(i))
		copy(inner, i)
		inner = append(inner, 0, 0)
		copy(inner[a+1:], inner[a:])
		copy(inner[b+1:], inner[b:])

		key := fmt.Sprintf("%v|%v|%v", inner, a, b)
		if hit, ok := cache[key]; !ok {
			for k := 0; k < t.dim[a]; k++ {
				inner[a], inner[b] = k, k
				if k == 0 {
					sum = t.f(inner...)
					continue
				}
				sum = t.t.Add(sum, t.f(inner...))
				profiler.Adds += 1
			}
			cache[key] = sum
			return sum
		} else {
			profiler.TraceCache += 1
			return hit
		}

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
		t.t,
	}, nil
}

func Product(t1, t2 Tensor, profiler *Profiler) Tensor {
	if profiler == nil {
		profiler = &Profiler{}
	}
	f := func(inner ...int) interface{} {
		i := make([]int, len(inner))
		copy(i, inner)
		// Assert they are the same type here
		// TODO(xam)
		profiler.Multiplies += 1
		return t1.t.Multiply(t1.f(i[0:len(t1.dim)]...),
			t2.f(i[len(t1.dim):]...))
	}

	t2dim := make([]int, len(t2.dim))
	copy(t2dim, t2.dim)
	return Tensor{
		f,
		t1.signature + t2.signature,
		append(t1.dim, t2dim...),
		t1.t,
	}
}

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
func (t Tensor) Reify() [][]interface{} {
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
	twoD := make([][]interface{}, contraDim)
	for i := 0; i < contraDim; i++ {
		twoD[i] = make([]interface{}, coDim)
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
