// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from fusion_msgs:msg/RadarPoints.idl
// generated code does not contain a copyright notice

#ifndef FUSION_MSGS__MSG__DETAIL__RADAR_POINTS__STRUCT_H_
#define FUSION_MSGS__MSG__DETAIL__RADAR_POINTS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'points'
#include "fusion_msgs/msg/detail/radar_point__struct.h"

/// Struct defined in msg/RadarPoints in the package fusion_msgs.
typedef struct fusion_msgs__msg__RadarPoints
{
  std_msgs__msg__Header header;
  fusion_msgs__msg__RadarPoint__Sequence points;
} fusion_msgs__msg__RadarPoints;

// Struct for a sequence of fusion_msgs__msg__RadarPoints.
typedef struct fusion_msgs__msg__RadarPoints__Sequence
{
  fusion_msgs__msg__RadarPoints * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} fusion_msgs__msg__RadarPoints__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // FUSION_MSGS__MSG__DETAIL__RADAR_POINTS__STRUCT_H_
