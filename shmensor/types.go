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
)

// This file is used to define some package-default Tensor types.
// Interface may become open in the future so clients can define their own.
// For example, Tensor spaces over finite fields, or modules over arbitrary rings.
// Integers.
type defaultInt struct{}

func (dt defaultInt) Multiply(x, y interface{}) interface{} {
	return x.(int) * y.(int)
}

func (dt defaultInt) Add(x, y interface{}) interface{} {
	return x.(int) + y.(int)
}

func NewIntTensor(f func(i ...int) int, signature string, dim []int) Tensor {
	return Tensor{
		func(i ...int) interface{} { return f(i...) },
		signature,
		dim,
		defaultInt{},
	}
}

// Reals.
type defaultReal struct{}

func (dt defaultReal) Multiply(x, y interface{}) interface{} {
	return x.(float64) * y.(float64)
}

func (dt defaultReal) Add(x, y interface{}) interface{} {
	return x.(float64) + y.(float64)
}

func NewRealTensor(f func(i ...int) float64, signature string, dim []int) Tensor {
	return Tensor{
		func(i ...int) interface{} { return f(i...) },
		signature,
		dim,
		defaultReal{},
	}
}

func NewRealFunction(f func(r float64) float64) Function {
	wrapper := func(i interface{}) interface{} {
		// Test type assertion and panic informatively first.
		return f(i.(float64))
	}
	return Function{
		wrapper,
		defaultReal{},
	}
}

// Complex numbers.
type defaultComplex struct{}

func (dt defaultComplex) Multiply(x, y interface{}) interface{} {
	return x.(complex128) * y.(complex128)
}

func (dt defaultComplex) Add(x, y interface{}) interface{} {
	return x.(complex128) + y.(complex128)
}

func NewComplexTensor(f func(i ...int) complex128, signature string, dim []int) Tensor {
	return Tensor{
		func(i ...int) interface{} { return f(i...) },
		signature,
		dim,
		defaultComplex{},
	}
}

// Strings
type defaultString struct{}

func (ds defaultString) Multiply(x, y interface{}) interface{} {
	return fmt.Sprintf("(%v)(%v)", x.(string), y.(string))
}

func (ds defaultString) Add(x, y interface{}) interface{} {
	return fmt.Sprintf("%v + %v", x.(string), y.(string))
}

func NewStringTensor(f func(i ...int) string, signature string, dim []int) Tensor {
	return Tensor{
		func(i ...int) interface{} { return f(i...) },
		signature,
		dim,
		defaultString{},
	}
}

func NewStringFunction(f func(s string) string) Function {
	wrapper := func(i interface{}) interface{} {
		// Test type assertion and panic informatively first.
		return f(i.(string))
	}
	return Function{
		wrapper,
		defaultString{},
	}
}
