# Tutorial for converting free function to templated callable

This tutorial helps users converting a free function to a templated callable that is acceptable by Zeus

---

## Step 1: Start with a Free Function

Here is a basic `double`-only function:

```cpp
double foo(double x) {
    return 0.5 * x * x;
}
```

This is simple, but inflexible. You can't run it on the GPU or with autodiff types.

---

## Step 2: Convert to a Templated Callable

To make it compatible with `Zeus`, convert it into a class with a templated call operator:

```cpp
// dimension of the objective
template <std::size_t DIM>
struct Foo {
  // templated call‐operator over any scalar type T
  template <typename T>
  __host__ __device__
  T operator()(const std::array<T, DIM>& a) const {
    // important not to introduce undefined behavior
    static_assert(N>=1,"function at least 2 dimensions");
    return T(0.5) * a[0] * a[0];
  }
};
```

This works with any `T` that supports basic arithmetic: `float`, `double`, `DualNumber` types.


