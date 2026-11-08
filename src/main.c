
#include "arena.h"
#include "base.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define _CRT_SECURE_NO_WARNINGS

typedef struct {
    u32 rows, cols;
    f32 *data;
} matrix;

matrix* mat_create(u32 rows, u32 cols, mem_arena *arena);
matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char *filename);
b32 mat_copy(matrix *dst, matrix *src);
void mat_clear(matrix *mat);
void mat_fill(matrix *mat, f32 val);
void mat_scale(matrix *mat, f32 scale);
f32 mat_sum(matrix *mat);
b32 mat_add(matrix *out, const matrix *a, const matrix *b);
b32 mat_sub(matrix *out, const matrix *a, const matrix *b);
b32 mat_mul(matrix *out, const matrix *a, const matrix *b, b8 zero_out,
            b8 transpose_a, b8 transpose_b);
b32 mat_relu(matrix *out, const matrix *in);
b32 mat_softmax(matrix *out, const matrix *in);
b32 mat_cross_entropy(matrix *out, const matrix *p, const matrix *q);
b32 mat_relu_add_grad(matrix *out, const matrix *in);
b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out);
b32 mat_cross_entropy_add_grad(matrix *out, const matrix *p, const matrix *q);

typedef enum {
    MV_FLAG_NONE = 0,
    
    MV_FLAG_REQUIRES_GRAD  = (1 << 0),
    MV_FLAG_PARAMETER      = (1 << 1),
    MV_FLAG_INPUT          = (1 << 2),
    MV_FLAG_OUTPUT         = (1 << 3),
    MV_FLAG_DESIRED_OUTPUT = (1 << 4),
    MV_FLAG_COST           = (1 << 5),
} model_var_flags;

typedef enum {
  MV_OP_NULL = 0,
  MV_OP_CREATE,

  _MV_OP_UNARY_START,

  MV_OP_RELU,
  MV_OP_SOFTMAX,

  _MV_OP_BINARY_START,

  MV_OP_ADD,
  MV_OP_SUB,
  MV_OP_MATMUL,
  MV_OP_CROSS_ENTROPY,
} model_var_ops;

#define MODEL_VAR_MAX_INPUTS 2
#define MV_NUM_INPUTS(op) ((op) < _MV_OP_BINARY_START ? 0 : ((op) < _MV_OP_UNARY_START ? 1 : 2))

typedef struct model_var {
  u32 index;
  u32 flags;

  matrix *val;
  matrix *grad;

  model_var_ops op;
  struct model_var *inputs[MODEL_VAR_MAX_INPUTS];
} model_var;

typedef struct {
  model_var **vars;
  u32 size;
} model_program;

typedef struct {
  u32 num_vars;

  matrix *input;
  matrix *output;
  matrix *desired_output;
  matrix *cost;

  model_program forward_program;
  model_program cost_program;
} model_context;

model_var *mv_create(
  mem_arena *arena, model_context *model,
  u32 rows, u32 cols, u32 flags
);

model_var *mv_relu(
  mem_arena *arena, model_context *model,
  model_var *input, u32 flags
);

model_var *mv_create(
  mem_arena *arena, model_context *model,
  u32 rows, u32 cols, u32 flags
);

model_var *mv_create(
  mem_arena *arena, model_context *model,
  u32 rows, u32 cols, u32 flags
);

model_var *mv_create(
  mem_arena *arena, model_context *model,
  u32 rows, u32 cols, u32 flags
);

void draw_mnist_digit(f32 *data);

int main() { 
    return 0;
}

void draw_mnist_digit(f32 *data) {
    for (u32 i = 0; i < 28; i++) {
        for (u32 j = 0; j < 28; j++) {
            f32 num = data[j + i * 28];
            u32 col = 232 + (u32)(num * 24);
            printf("\x1b[48;5;%dm  ", col);
        }
        printf("\n");
    }
    printf("\x1b[0m");    
}

matrix *mat_create(u32 rows, u32 cols, mem_arena *arena) {
    matrix *mat = PUSH_STRUCT(arena, matrix);
    mat->rows = rows;
    mat->cols = cols;
    mat->data = PUSH_ARRAY(arena, f32, (u64)rows * cols);

    return mat;
}

matrix *mat_load(mem_arena *arena, u32 rows, u32 cols, const char *filename) {
    matrix *mat = mat_create(rows, cols, arena);

    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return NULL;
    }
    
    fseek(f, 0, SEEK_END);
    u64 size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size = MIN(size, sizeof(f32) * rows * cols);
    fread(mat->data, 1, size, f);
    
    fclose(f);
    
    return mat;
}

b32 mat_copy(matrix *dst, matrix *src) {
    if (dst->rows != src->rows || dst->cols != src->cols) {
        return false;
    }
    memcpy(dst->data, src->data, sizeof(f32) * (u64)dst->rows * dst->cols);

    return true;
}
void mat_clear(matrix *mat) {
    memset(mat->data, 0, sizeof(f32) * (u64)mat->rows * mat->cols);
}

void mat_fill(matrix *mat, f32 val) {
    u64 size = (u64)mat->rows * mat->cols;

    for (u64 i = 0; i < size; i++) {
        mat->data[i] = val;
    }
}

void mat_scale(matrix *mat, f32 scale) {
    u64 size = (u64)mat->rows * mat->cols;

    for (u64 i = 0; i < size; i++) {
        mat->data[i] *= scale;
    }
}

f32 mat_sum(matrix *mat) {
    u64 size = (u64)mat->rows * mat->cols;
    f32 sum = 0.0f;

    for (u64 i = 0; i < size; i++) {
        sum += mat->data[i];
    }

    return sum;
}

b32 mat_add(matrix *out, const matrix *a, const matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return false;
    }
    u64 size = (u64)a->rows * a->cols;

    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return true;
}

b32 mat_sub(matrix *out, const matrix *a, const matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return false;
    }
    u64 size = (u64)a->rows * a->cols;

    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }

    return true;
}

void _mat_mul_nn(matrix *out, const matrix *a, const matrix *b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 k = 0; k < a->cols; k++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[i + j * out->cols] +=
                    a->data[k + i * a->cols] * b->data[j + k * b->cols];
            }
        }
    }
}

void _mat_mul_nt(matrix *out, const matrix *a, const matrix *b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            for (u64 k = 0; k < a->cols; k++) {
                out->data[j + i * out->cols] +=
                    a->data[k + i * a->cols] * b->data[k + j * b->cols];
            }
        }
    }
}

void _mat_mul_tn(matrix *out, const matrix *a, const matrix *b) {
    for (u64 k = 0; k < a->rows; k++) {
        for (u64 i = 0; i < out->rows; i++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] +=
                    a->data[i + k * a->cols] * b->data[j + k * b->cols];
            }
        }
    }
}

void _mat_mul_tt(matrix *out, const matrix *a, const matrix *b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            for (u64 k = 0; k < a->cols; k++) {
                out->data[i + j * out->cols] +=
                    a->data[i + k * a->cols] * b->data[k + j * b->cols];
            }
        }
    }
}

b32 mat_mul(matrix *out, const matrix *a, const matrix *b, b8 zero_out,
            b8 transpose_a, b8 transpose_b) {
    u32 a_rows = transpose_a ? a->cols : a->rows;
    u32 a_cols = transpose_a ? a->rows : a->cols;
    u32 b_rows = transpose_a ? b->cols : b->rows;
    u32 b_cols = transpose_a ? b->rows : b->cols;

    if (a_cols != b_rows) {
        return false;
    }
    if (out->rows != a_rows || out->cols != b_cols) {
        return false;
    }

    if (zero_out) {
        mat_clear(out);
    }

    u32 tranpose = (transpose_a << 1) | transpose_b;
    switch (tranpose) {
    case 0b00: {
        _mat_mul_nn(out, a, b);
    } break;
    case 0b01: {
        _mat_mul_nt(out, a, b);
    } break;
    case 0b10: {
        _mat_mul_tn(out, a, b);
    } break;
    case 0b11: {
        _mat_mul_tt(out, a, b);
    } break;
    }

    return true;
}

b32 mat_relu(matrix *out, const matrix *in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = MAX(0, in->data[i]);
    }

    return true;
}

b32 mat_softmax(matrix *out, const matrix *in) {
    // o_i = e^a_i / sum(e^a_i)
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    u64 size = (u64)in->rows * in->cols;

    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = expf(in->data[i]);
        sum += out->data[i];
    }

    mat_scale(out, 1.0f / sum);

    return true;
}

b32 mat_cross_entropy(matrix *out, const matrix *p, const matrix *q) {
    if (p->rows != q->rows || p->cols != q->cols) {
        return false;
    }
    if (out->rows != p->rows || out->cols != p->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] =
            p->data[i] == 0.0f ? 0.0f : p->data[i] * -logf(q->data[i]);
    }

    return true;
}
b32 mat_relu_add_grad(matrix *out, const matrix *in);
b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out);
b32 mat_cross_entropy_add_grad(matrix *out, const matrix *p, const matrix *q);
