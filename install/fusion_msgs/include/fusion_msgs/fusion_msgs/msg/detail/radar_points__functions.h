// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from fusion_msgs:msg/RadarPoints.idl
// generated code does not contain a copyright notice

#ifndef FUSION_MSGS__MSG__DETAIL__RADAR_POINTS__FUNCTIONS_H_
#define FUSION_MSGS__MSG__DETAIL__RADAR_POINTS__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "fusion_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "fusion_msgs/msg/detail/radar_points__struct.h"

/// Initialize msg/RadarPoints message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * fusion_msgs__msg__RadarPoints
 * )) before or use
 * fusion_msgs__msg__RadarPoints__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
bool
fusion_msgs__msg__RadarPoints__init(fusion_msgs__msg__RadarPoints * msg);

/// Finalize msg/RadarPoints message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
void
fusion_msgs__msg__RadarPoints__fini(fusion_msgs__msg__RadarPoints * msg);

/// Create msg/RadarPoints message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * fusion_msgs__msg__RadarPoints__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
fusion_msgs__msg__RadarPoints *
fusion_msgs__msg__RadarPoints__create();

/// Destroy msg/RadarPoints message.
/**
 * It calls
 * fusion_msgs__msg__RadarPoints__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
void
fusion_msgs__msg__RadarPoints__destroy(fusion_msgs__msg__RadarPoints * msg);

/// Check for msg/RadarPoints message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
bool
fusion_msgs__msg__RadarPoints__are_equal(const fusion_msgs__msg__RadarPoints * lhs, const fusion_msgs__msg__RadarPoints * rhs);

/// Copy a msg/RadarPoints message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
bool
fusion_msgs__msg__RadarPoints__copy(
  const fusion_msgs__msg__RadarPoints * input,
  fusion_msgs__msg__RadarPoints * output);

/// Initialize array of msg/RadarPoints messages.
/**
 * It allocates the memory for the number of elements and calls
 * fusion_msgs__msg__RadarPoints__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
bool
fusion_msgs__msg__RadarPoints__Sequence__init(fusion_msgs__msg__RadarPoints__Sequence * array, size_t size);

/// Finalize array of msg/RadarPoints messages.
/**
 * It calls
 * fusion_msgs__msg__RadarPoints__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
void
fusion_msgs__msg__RadarPoints__Sequence__fini(fusion_msgs__msg__RadarPoints__Sequence * array);

/// Create array of msg/RadarPoints messages.
/**
 * It allocates the memory for the array and calls
 * fusion_msgs__msg__RadarPoints__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
fusion_msgs__msg__RadarPoints__Sequence *
fusion_msgs__msg__RadarPoints__Sequence__create(size_t size);

/// Destroy array of msg/RadarPoints messages.
/**
 * It calls
 * fusion_msgs__msg__RadarPoints__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
void
fusion_msgs__msg__RadarPoints__Sequence__destroy(fusion_msgs__msg__RadarPoints__Sequence * array);

/// Check for msg/RadarPoints message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
bool
fusion_msgs__msg__RadarPoints__Sequence__are_equal(const fusion_msgs__msg__RadarPoints__Sequence * lhs, const fusion_msgs__msg__RadarPoints__Sequence * rhs);

/// Copy an array of msg/RadarPoints messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
bool
fusion_msgs__msg__RadarPoints__Sequence__copy(
  const fusion_msgs__msg__RadarPoints__Sequence * input,
  fusion_msgs__msg__RadarPoints__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // FUSION_MSGS__MSG__DETAIL__RADAR_POINTS__FUNCTIONS_H_
