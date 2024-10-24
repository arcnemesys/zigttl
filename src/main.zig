//! By convention, main.zig is where your main function lives in the case that
//! you are building an executable. If you are making a library, the convention
//! is to delete this file and start with root.zig instead.
const std = @import("std");

// Reference material:
// Image manipulation: https://pedropark99.github.io/zig-book/Chapters/13-image-filter.html
// Matrix multiplication: https://svaniksharma.github.io/posts/2023-05-07-optimizing-matrix-multiplication-with-zig/
//
// DNN in Zig: https://monadmonkey.com/dnns-from-scratch-in-zig
//
// zig-neural-network repo: https://github.com/MadLittleMods/zig-neural-networks
// Demistifying strides: https://martinlwx.github.io/en/how-to-reprensent-a-tensor-or-ndarray/
//
// Guide to shape and strides: https://ajcr.net/stride-guide-part-1/
// Some tasks, such as changing an image to grayscale, can be programmed with a
// predetermined set of instructions, that yield a deterministic output.
//
//
// Other tasks, can not be handled this way, such as determining if the image contains
// a cat or a dog, and returning a string with the correct answer.
//
// There's no specific set of instructions to solve this, but it can be solved:
//

// fn cat_or_dog(input: []f32, w1: []f32, w2: []f32) i32 {
//     const x1 = matrix_multiplication(input, w1);
//     const x2 = relu(x1);
//     const x3 = matrix_multiplication(x2, w2);
//     const x4 = logsoftmax(x3);
//     if (x4[0] < x4[1]) {
//         return 0;
//     } else {
//         return 1;
//     }
// }

// Noticeably, this function takes two additional arrays, w1, and w2, apart from the image.
// The function then multiplies the input array by the w1 array.
// `relu` is a mathematical operation, equivalent to `max(x, 0)` for every element
// of the array.
// `logsoftmax` is another mathematical operation.
//
// For specific values of w1, and w1, the function will give the correct output for 99% of
// images: defining these functions and finding the optimal value of w1 and w2
// is effectively what deep learning is about.
//
// Such functions are called neural networks, with w1, and w2 being parameters or weights.
// Training a nueral network is how the optimal values of w1, and w2 are found.
//
// ChatGPT & co are powered by these networks, and generate responses to messages, by
// converting messages into numbers, operating on them, and the parameters, to output
// the next word, until the response is generated.
//
const MAX_PREVS: u8 = 3;
const MAX_ARGS: u8 = 5;
const MAX_PARAM_TENSORS: u8 = 10;

const MAT_MUL: u8 = 0;
const MEAN: u8 = 1;
const MUL: u8 = 2;
const RELU: u8 = 3;
const LOG_SOFT_MAX: u8 = 4;

pub const Arr = struct { 
    // Slice that stores the actual contents of the array
    values: []f32, 
    // Slice storing the sice of each dimension
    shape: []usize, 
    // Slice containing the stride for each dimension
    strides: []usize,
    // Number of dimensions
    ndim: usize, 
    // Total number of elements in the array
    size: usize, 

    pub fn deinit(self: *Arr, allocator: std.mem.Allocator) void {
        allocator.free(self.shape);
        allocator.free(self.strides);
        allocator.free(self.values);
        allocator.destroy(self);
    }
};

// Everything we need is in this struct. The values are stored in `values`, a 1D array.
// Shape, represents the size and dimension of the tensor, so given a 3D tensor with
// dimensions 2x3x4, we have 2 matrices, with each matrix having 3 rows, and 4 columns, so
// that the shape is [2, 3, 4]

fn mat_mul(a: *Arr, b: *Arr, c: *Arr) *Tensor {
    const p: i32 = a.shape[0];
    const q: i32 = a.shape[1];
    const r = b.shape[1];

    const i: usize = 0;
    const j: usize = 0;
    const k: usize = 0;
    while (i < p) {
        while (j < r) {
            var temp: f32 = 0.0;
            while (k < q) {
                const pos_a = i * a.strides[0] + k * a.strides[1];
                const pos_b = k * b.strides[0] + j * b.strides[1];
                temp += a.values[pos_a] * b.values[pos_b];
                k += 1;
            }
            const pos_c = i * c.strides[0] + j * c.strides[1];
            c.values[pos_c] = temp;
            j += 1;
        }
        i += 1;
    }
}

fn create_arr(allocator: std.mem.Allocator, length: usize) ?*Arr {
    const arr = allocator.create(Arr) catch return null;
    errdefer allocator.destroy(arr);

    arr.* = Arr{
        .data = allocator.alloc(i32, length) catch {
            allocator.destroy(arr);
            return null;
        },
        .length = length,
    };

    return arr;
}
fn create_arr_zeros(allocator: std.mem.Allocator, shape: []const usize) !*Arr {
    // Allocate memory for the array
    var arr = try allocator.create(Arr);
    // If that fails, an error will be returned from the function,
    // In the event of subsequent allocation fails, this ensures arr is destroyed.
    errdefer allocator.destroy(arr);

    // Set the number of dimensions to the input shape.
    arr.ndim = shape.len;
    // Allocate memory for the input shape, and copy it.
    arr.shape = try allocator.dupe(usize, shape);

    // If that fails, an error will be returned and arr, arr.shape will be freed.
    // Ensures that arr.shape will be freed if allocation fails.
    errdefer allocator.free(arr.shape);
    
    arr.strides = try allocator.alloc(usize, arr.ndim);
    errdefer allocator.free(arr.strides);

    arr.size = 1; // Initialize total size to one
    var i: usize = arr.ndim; // Start from the last dimension

    while (i > 0) {
        // Loop through dimensions from last to to first.
        i -= 1;
        // Set the stride for this dimension
        arr.strides[i] = arr.size;
        // Update the total size.
        arr.size *= shape[i];

        // arr.size is 1, and i is 2, which is the last dimension, of three dimensions,
        // due to zero-indexing.
        // We subtract 1 from i, giving us two, and then we set arr.strides[i] to arr.size,
        // or 1.
        // Now the stride for the dimension we're on, is 1
    }

    // Allocate memory for the values.
    arr.values = try allocator.alloc(f32, arr.size);
    // Ensure that arr.values is freed if allocation fails.
    errdefer allocator.free(arr.values);
    std.mem.set(f32, arr.values, 0);

    return arr;
// The reason for looping from the last dimension, to the first when calculating
// strides for multi-dim arrays, is because of how they're generally laid out in memory.
//
// This is related to `row-major order`, which is the standard memory layout for multi-dim 
// arrays in many languages.
//
// Memory layout: in `row-major order`, elements that are adjacent in the last dimension,
// are stored continguously.
//
// Stride: The stride of a dimension indicates how many elements to skip in the one dimensional
// memory layout, to move one step in that dimension.
//
// The stride for the last dimension is always one, and for each dimension, as we move left,
// the stride is the product of all of the dimension sizes to its right.

// Given a 3D array with shape [2,3,4]
// arr.size = 1
// i = ndim, and ndim = 2, since we have dim 0, dim 1, and dim 2
// arr.strides has the same number of elements as ndim, i.e 3 elements.
// arr.strides[i] == arr.strides[2] = 1, 
}
// fn create_zero_tensor(shape: *[]i32, ndim: i32) void {
    // const d: *Arr = create_arr_zeros(shape, ndim);
    // var tensor: *Tensor = ()
// }
fn mat_mul2d(a: *Tensor, b: *Tensor) *Tensor {
    const p: i32 = a.data.shape[0];
    const q: i32 = a.data.shape[1];
    const r = b.data.shape[1];

    // tensor: *Tensor = create_zero_tensor()
}
fn mat_mul_backward()

// Our input, w1, and w2 are tensors, and we need to find specific values of w1 and w2
// that make the function work.
// This requires data containing accurate inputs and outputs, and an additional function
// that operates on the outputs of the network, and the labelled ones, and returns
// a score representing how good the network is.

fn loss_fn(logsoftmax_outputs: Arr, labels: Arr, a: Arr) f32 {
    var s: f32 = 0.0;
    for (0..logsoftmax_outputs.size) |i| {
        s += a.data[i] * labels.data[i] * -1;
    }

    return s / a.size;
}

const Arg = union {
    ival: i32,
    fval: f32,
};

const Tensor = struct {
    data: *Arr,
    // Stores the gradient, an ND array of the same shape as `data`.
    grad: *Arr,
    // Stores the operation used to create the tensor.
    op: i32,
    // An array of pointers to other tensors processed by op
    prevs: [MAX_PREVS]?*Tensor,
    num_prevs: i32,
    args: [MAX_ARGS]Arg,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);

        self.* = .{
            .data = undefined,
            .grad = undefined,
            .op = 0,
            .prevs = [_]?*Tensor{null} ** MAX_PREVS,
            .num_prevs = 0,
            .args = undefined,
        };
        return self;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.destroy(self);
    }
};

fn mean_backward(out: *Tensor) void {
    for (0..out.prevs[0].grad.size) |i| {
        out.prevs[0].grad.values[i] += out.grad.values;
    }
}

fn mul_backward(out: *Tensor) void {
    for (0..out.data.size) |i| {
        out.prevs[0].grad.values[i] += out.grad.values;
        out.prevs[1].grad.values[i] += out.grad.values;
    }
}

fn backward(tensor: *Tensor) void {
    if (tensor.op == MUL) {
        mul_backward(tensor);
    } else if (tensor.op == MEAN) {
        mean_backward(tensor);
    }

    for (0..tensor.num_prevs) |i| {
        backward(tensor.prevs[i]);
    }
}
// The closer the loss value is to zero, the better the values of w1 and w2, and thus the
// better the nn function is, so finding good values for w1 and w2, is the same as finding
// values for them that produce low values for the loss function.
pub fn main() !void {
    // Prints to stderr (it's a shortcut based on `std.io.getStdErr()`)
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});

    // stdout is for the actual output of your application, for example if you
    // are implementing gzip, then only the compressed bytes should be sent to
    // stdout, not any debugging messages.
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    try stdout.print("Run `zig build test` to run the tests.\n", .{});

    try bw.flush(); // Don't forget to flush!
}

test "mat_mul" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};

    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const vector = Arr{ .values = try allocator.alloc(f32, 5), .shape = try allocator.alloc(i32, 1), .strides = try allocator.alloc(i32, 1), .ndim = 1, .size = 5 };
    defer allocator.free(vector.values);
    defer allocator.free(vector.shape);
    defer allocator.free(vector.strides);

    vector.shape[0] = 5;
    vector.strides[0] = 1;

    const matrix = Arr{
        .values = try allocator.alloc(f32, 6),
        .shape = try allocator.alloc(i32, 2),
        .strides = try allocator.alloc(i32, 2),
        .ndim = 2,
        .size = 6,
    };
    defer allocator.free(matrix.values);
    defer allocator.free(matrix.shape);
    defer allocator.free(matrix.strides);

    matrix.shape[0] = 2;
    matrix.shape[1] = 3;
    matrix.strides[0] = 3;
    matrix.strides[1] = 1;

    // 3D tensor
    const tensor3d = Arr{
        .values = try allocator.alloc(f32, 24),
        .shape = try allocator.alloc(i32, 3),
        .strides = try allocator.alloc(i32, 3),
        .ndim = 3,
        .size = 24,
    };
    defer allocator.free(tensor3d.values);
    defer allocator.free(tensor3d.shape);
    defer allocator.free(tensor3d.strides);

    tensor3d.shape[0] = 2;
    tensor3d.shape[1] = 3;
    tensor3d.shape[2] = 4;
    tensor3d.strides[0] = 12;
    tensor3d.strides[1] = 4;
    tensor3d.strides[2] = 1;

    // Print some information
    std.debug.print("Vector: ndim={}, size={}\n", .{ vector.ndim, vector.size });
    std.debug.print("Matrix: ndim={}, size={}\n", .{ matrix.ndim, matrix.size });
    std.debug.print("3D Tensor: ndim={}, size={}\n", .{ tensor3d.ndim, tensor3d.size });
}
test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // Try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
    const input_bytes = std.testing.fuzzInput(.{});
    try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input_bytes));
}
