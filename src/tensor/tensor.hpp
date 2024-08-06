#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <memory>
#include <functional>
#include <algorithm>

namespace tensorbarrel
{
    template <typename TValue>
    class Tensor
    {
    public:
        using ValueType = TValue;
        using SizeValueType = std::size_t;
        using IndexValueType = std::size_t;
        using ShapeType = std::vector<SizeValueType>;
        using IndexType = std::vector<IndexValueType>;

        // TODO: Allow the function to accept the index of the element
        using allocator_type = std::function<ValueType()>;

        Tensor();
        ~Tensor() = default;
        Tensor(const ShapeType &shape);
        Tensor(const Tensor<TValue> &other);
        Tensor(Tensor<TValue> &&other);

        friend void swap(Tensor<TValue> &first, Tensor<TValue> &second)
        {
            using std::swap;
            swap(first.m_shape, second.m_shape);
            swap(first.m_capacity, second.m_capacity);
            swap(first.m_data, second.m_data);
        }

        Tensor<TValue> &operator=(Tensor<TValue> other);
        Tensor<TValue> &operator=(Tensor<TValue> &&other);

        // Static Constructors
        static Tensor<TValue> Zeros(const ShapeType &shape);
        static Tensor<TValue> Ones(const ShapeType &shape);
        static Tensor<TValue> RandomInt(const ShapeType &shape);
        static Tensor<TValue> RandomFloat(const ShapeType &shape);
        static Tensor<TValue> Allocator(const ShapeType &shape, allocator_type allocator);

        // Fill
        void Fill(const ValueType &value);
        void Fill(allocator_type allocator);

        // Reshape
        void Reshape(const ShapeType &shape, const ValueType &value);
        void Reshape(const ShapeType &shape, const allocator_type &allocator);

        // Get specific element
        ValueType &operator[](const IndexType &index);
        const ValueType &operator[](const IndexType &index) const;

        // Get Tensor Slice
        Tensor<TValue> operator[](IndexValueType index);
        const Tensor<TValue> operator[](IndexValueType index) const;

        // Get SubTensor
        Tensor<TValue> operator()(const IndexType &start, const IndexType &end);

        // Get Shape
        const ShapeType &Shape() const { return m_shape; };
        // Get Capacity
        SizeValueType Capacity() const { return m_capacity; };

        // Comparison
        bool operator==(const Tensor<TValue> &other) const;
        bool operator!=(const Tensor<TValue> &other) const;

        // Math Operations
        Tensor<TValue> operator+(const Tensor<TValue> &other) const;
        Tensor<TValue> operator-(const Tensor<TValue> &other) const;
        Tensor<TValue> operator*(const Tensor<TValue> &other) const;
        Tensor<TValue> operator/(const Tensor<TValue> &other) const;

        Tensor<TValue> &operator+=(const Tensor<TValue> &other);
        Tensor<TValue> &operator-=(const Tensor<TValue> &other);
        Tensor<TValue> &operator*=(const Tensor<TValue> &other);
        Tensor<TValue> &operator/=(const Tensor<TValue> &other);

        Tensor<TValue> operator+(const ValueType &value) const;
        Tensor<TValue> operator-(const ValueType &value) const;
        Tensor<TValue> operator*(const ValueType &value) const;
        Tensor<TValue> operator/(const ValueType &value) const;

        Tensor<TValue> &operator+=(const ValueType &value);
        Tensor<TValue> &operator-=(const ValueType &value);
        Tensor<TValue> &operator*=(const ValueType &value);
        Tensor<TValue> &operator/=(const ValueType &value);

        // Stores the result in the result tensor
        Tensor<ValueType> MatMul(const Tensor<TValue> &other) const;
        // Stores the result in the current tensor
        void MatMul(const Tensor<TValue> &other);

        // Transpose
        Tensor<ValueType> &Transposed(ShapeType &axis_order);
        Tensor<ValueType> Transpose(ShapeType &axis_order) const;

        // TODO: Iterator classes for the tensor

    private:
        std::shared_ptr<ValueType> m_data;
        ShapeType m_shape;
        SizeValueType m_capacity;
    };
}

#endif /* TENSOR_HPP */
