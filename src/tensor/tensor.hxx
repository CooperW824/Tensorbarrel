#include "tensor.hpp"

#include <random>

template <typename TValue>
tensorbarrel::Tensor<TValue>::Tensor() : m_shape({0}), m_capacity(0), m_data(nullptr) {}

template <typename TValue>
tensorbarrel::Tensor<TValue>::Tensor(const ShapeType &shape) : m_shape(shape), m_capacity(1)
{
    for (const auto &s : m_shape)
    {
        m_capacity *= s;
    }
    m_data = std::make_shared<TValue[]>(m_capacity);
}

template <typename TValue>
tensorbarrel::Tensor<TValue>::Tensor(const tensorbarrel::Tensor<TValue> &other) : m_shape(other.m_shape), m_capacity(other.m_capacity)
{
    m_data = std::make_shared<TValue[]>(m_capacity);
    std::copy(other.m_data.get(), other.m_data.get() + m_capacity, m_data.get());
}

template <typename TValue>
inline tensorbarrel::Tensor<TValue>::Tensor(tensorbarrel::Tensor<TValue> &&other)
{
    m_shape = std::move(other.m_shape);
    m_capacity = std::move(other.m_capacity);
    m_data = std::move(other.m_data);
}

template <typename TValue>
inline tensorbarrel::Tensor<TValue> &tensorbarrel::Tensor<TValue>::operator=(tensorbarrel::Tensor<TValue> other)
{
    swap(*this, other);
    return *this;
}

template <typename TValue>
inline tensorbarrel::Tensor<TValue> &tensorbarrel::Tensor<TValue>::operator=(tensorbarrel::Tensor<TValue> &&other)
{
    if (this == &other)
        return *this;

    m_shape = std::move(other.m_shape);
    m_capacity = std::move(other.m_capacity);
    m_data = std::move(other.m_data);
    return *this;
}

template <typename TValue>
inline tensorbarrel::Tensor<TValue> tensorbarrel::Tensor<TValue>::Zeros(const ShapeType &shape)
{
    tensorbarrel::Tensor<TValue> tensor(shape);
    tensor.Fill(0);
    return tensor;
}

template <typename TValue>
inline tensorbarrel::Tensor<TValue> tensorbarrel::Tensor<TValue>::Ones(const ShapeType &shape)
{
    tensorbarrel::Tensor<TValue> tensor(shape);
    tensor.Fill(1);
    return tensor;
}

template <typename TValue>
inline tensorbarrel::Tensor<TValue> tensorbarrel::Tensor<TValue>::RandomInt(const ShapeType &shape)
{
    tensorbarrel::Tensor<TValue> tensor(shape);
    tensor.Fill([]()
                { return static_cast<TValue>(std::rand()); });
    return tensor;
}

template <typename TValue>
inline tensorbarrel::Tensor<TValue> tensorbarrel::Tensor<TValue>::RandomFloat(const ShapeType &shape)
{
    tensorbarrel::Tensor<TValue> tensor(shape);
    tensor.Fill([]()
                { return static_cast<TValue>(std::rand()) / static_cast<TValue>(RAND_MAX); });
    return tensor;
}

template <typename TValue>
inline tensorbarrel::Tensor<TValue> tensorbarrel::Tensor<TValue>::Allocator(const ShapeType &shape, allocator_type allocator)
{
    tensorbarrel::Tensor<TValue> tensor(shape);
    tensor.Fill(allocator);
    return tensor;
}

template <typename TValue>
inline void tensorbarrel::Tensor<TValue>::Fill(const ValueType &value)
{
    std::fill(m_data.get(), m_data.get() + m_capacity, value);
}

template <typename TValue>
inline void tensorbarrel::Tensor<TValue>::Fill(allocator_type allocator)
{

    // TODO: Allow the function to accept the index of the element
    for (SizeValueType i = 0; i < m_capacity; ++i)
    {
        m_data[i] = allocator();
    }
}

template <typename TValue>
inline void tensorbarrel::Tensor<TValue>::Reshape(const ShapeType &shape, const ValueType &value)
{
    // TODO: Implement Reshape
    // This function should reshape the tensor to the new shape copying the old data and 
    // retaining the shape of the old data in the new tensor if possible. Fill any new spots with the value
    // If the shape is smaller than the original shape, the data should be truncated but should do its best to retain the original shape
}
