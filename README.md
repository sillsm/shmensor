# shmensor
Tiny tensor library in Go.

Most tensor operations are just combinations of a) tensor products and b) contractions (sums) over the tensors where 2 of the n indices are fixed and range 0 to their dimension. So I implement those two functions, and then provide multiple annotated examples this happening. For example, matrix multiplication (a rows b columns times b rows c columns) is actually a tensor product, creating 4 indices, and then an inner index contraction. Matrix times vector is the same, but it contracts from a 3 tensor back to a column vector.

I picked a canonical format that makes tensors easier to see. I pretend each tensor is the Kronecker product of a bunch of smaller tensors. They are represented as two dimensional matrices. I take the global row column indices, and trade them in for a list of coordinates with the contravariance (up and downness) going up to down, and the covariance (left to rightness) going left to right.

TODO:
1. Index juggling. Going to designate metric tensors with an additional Boolean.
2. Allow transpose by representing indices out of alphabetical order.


References:

* Here's a great beginner resource for abstract index notation: https://en.wikipedia.org/wiki/Abstract_index_notation

* Contravariance and covariance: https://en.wikipedia.org/wiki/Covariance_and_contravariance_of_vectors

* Pavel Grinfeld's lectures on Youtube are essential viewing: https://www.youtube.com/watch?v=e0eJXttPRZI
