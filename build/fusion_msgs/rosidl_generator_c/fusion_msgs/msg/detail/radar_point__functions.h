// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from fusion_msgs:msg/RadarPoint.idl
// generated code does not contain a copyright notice

#ifndef FUSION_MSGS__MSG__DETAIL__RADAR_POINT__FUNCTIONS_H_
#define FUSION_MSGS__MSG__DETAIL__RADAR_POINT__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "fusion_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "fusion_msgs/msg/detail/radar_point__struct.h"

/// Initialize msg/RadarPoint message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * fusion_msgs__msg__RadarPoint
 * )) before or use
 * fusion_msgs__msg__RadarPoint__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
bool
fusion_msgs__msg__RadarPoint__init(fusion_msgs__msg__RadarPoint * msg);

/// Finalize msg/RadarPoint message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
void
fusion_msgs__msg__RadarPoint__fini(fusion_msgs__msg__RadarPoint * msg);

/// Create msg/RadarPoint message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * fusion_msgs__msg__RadarPoint__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
fusion_msgs__msg__RadarPoint *
fusion_msgs__msg__RadarPoint__create();

/// Destroy msg/RadarPoint message.
/**
 * It calls
 * fusion_msgs__msg__RadarPoint__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
void
fusion_msgs__msg__RadarPoint__destroy(fusion_msgs__msg__RadarPoint * msg);

/// Check for msg/RadarPoint message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
bool
fusion_msgs__msg__RadarPoint__are_equal(const fusion_msgs__msg__RadarPoint * lhs, const fusion_msgs__msg__RadarPoint * rhs);

/// Copy a msg/RadarPoint message.
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
fusion_msgs__msg__RadarPoint__copy(
  const fusion_msgs__msg__RadarPoint * input,
  fusion_msgs__msg__RadarPoint * output);

/// Initialize array of msg/RadarPoint messages.
/**
 * It allocates the memory for the number of elements and calls
 * fusion_msgs__msg__RadarPoint__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
bool
fusion_msgs__msg__RadarPoint__Sequence__init(fusion_msgs__msg__RadarPoint__Sequence * array, size_t size);

/// Finalize array of msg/RadarPoint messages.
/**
 * It calls
 * fusion_msgs__msg__RadarPoint__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
void
fusion_msgs__msg__RadarPoint__Sequence__fini(fusion_msgs__msg__RadarPoint__Sequence * array);

/// Create array of msg/RadarPoint messages.
/**
 * It allocates the memory for the array and calls
 * fusion_msgs__msg__RadarPoint__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
fusion_msgs__msg__RadarPoint__Sequence *
fusion_msgs__msg__RadarPoint__Sequence__create(size_t size);

/// Destroy array of msg/RadarPoint messages.
/**
 * It calls
 * fusion_msgs__msg__RadarPoint__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
void
fusion_msgs__msg__RadarPoint__Sequence__destroy(fusion_msgs__msg__RadarPoint__Sequence * array);

/// Check for msg/RadarPoint message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_fusion_msgs
bool
fusion_msgs__msg__RadarPoint__Sequence__are_equal(const fusion_msgs__msg__RadarPoint__Sequence * lhs, const fusion_msgs__msg__RadarPoint__Sequence * rhs);

/// Copy an array of msg/RadarPoint messages.
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
fusion_msgs__msg__RadarPoint__Sequence__copy(
  const fusion_msgs__msg__RadarPoint__Sequence * input,
  fusion_msgs__msg__RadarPoint__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // FUSION_MSGS__MSG__DETAIL__RADAR_POINT__FUNCTIONS_H_
