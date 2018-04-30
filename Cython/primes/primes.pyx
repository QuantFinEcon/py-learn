def primes(int nb_primes):

    cdef int n, i, len_p

    cdef int p[1000]
    if nb_primes > 1000:
        nb_primes = 1000

    len_p = 0  # The current number of elements in p.
    n = 2
    while len_p < nb_primes:
        # Is n prime?
        """ Because no Python objects are referred to,
         the loop is translated entirely into C code, 
         and thus runs very fast. You will notice the
          way we iterate over the p C array. """
        for i in p[:len_p]:
            if n % i == 0:
                break

        # If no break occurred in the loop, we have a prime.
        else:
            p[len_p] = n
            len_p += 1
        n += 1

    # Let's return the result in a python list:
    # copy C array into python list
    # Cython can automatically convert many C types from and to Python types
    result_as_list  = [prime for prime in p[:len_p]]
    return result_as_list