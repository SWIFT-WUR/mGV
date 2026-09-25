# Optimizing GPU performance

For mGV we want to be able to run the model as fast as possible on GPUs.
What performs well and what doesn't is not always clear initially.

Whenever you make changes to the model code, you should do a test run with a larger dataset
on a machine with a GPU, and see if the performance is impacted.
Running smaller datasets or on CPU does not give you any insight into this.

On this page are some quick takeaways to keep in mind when writing code for the model.

## Moving data to GPU

- Don't initialize empty arrays on GPU to then fill them with data, use Adapt.@adapt instead to move data to GPU.
- When updating existing arrays on GPU, use copyto! and avoid slice notation.

## Writing kernels

- KernelAbstractions' `@Const` means that ***during the kernel*** this input does not change. NOT that it never changes.
- Put as many operations in a kernel as possible. Do not e.g., accumulate some variables separately before passing them into the kernel. Performing the accumulation inside the kernel will lead to vastly higher performance.
