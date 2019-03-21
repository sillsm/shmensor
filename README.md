# shmensor
Tiny tensor library in Go. For demonstration and education.

I've been struggling with tensors for years, and writing this small program has helped me solidify my understanding. I hope playing with it can do the same for you.

Most tensor operations are just combinations of a) tensor products and b) matrix contractions. So I implement those two functions, and then provide multiple annotated examples in demo.go of this happening over and over. For example, matrix multiplication is actually a tensor product, creating 4 indices, and then an inner index contraction. Matrix times vector is the same, but it creates contracts from a 3 tensor back to a column vector.

It's hard to visualize tensor components without tricking yourself into think they are matrices. Tensors are not matrices. They're not even just "multi-dimensional matrices". Still, I picked a canonical format that makes them easier to see. I pretend each tensor is the Kronecker product of a bunch of smaller tensors. They are represented as two dimensional matrices. I take the global row column indices, and trade them in for a list of coordinates with the contravariance (up and downness) going up to down, and the covariance (left to rightness) going left to right. Look at the examples; the really long and fat tensors can help develop a more intuitive understanding of contra and covariance. The numbering system is cool but poorly documented for now, see the getCoordinates function.

TODO:
1. Index juggling. Going to designate metric tensors with an additional Boolean.
2. Allow transpose by representing indices out of alphabetical order.
3. TESTS.


References:

* Here's a great beginner resource for abstract index notation: https://en.wikipedia.org/wiki/Abstract_index_notation

* Contravariance and covariance: https://en.wikipedia.org/wiki/Covariance_and_contravariance_of_vectors

* Pavel Grinfeld's lectures on Youtube are essential viewing: https://www.youtube.com/watch?v=e0eJXttPRZI
