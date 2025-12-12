// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from fusion_msgs:msg/RadarPoint.idl
// generated code does not contain a copyright notice

#ifndef FUSION_MSGS__MSG__DETAIL__RADAR_POINT__STRUCT_H_
#define FUSION_MSGS__MSG__DETAIL__RADAR_POINT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/RadarPoint in the package fusion_msgs.
typedef struct fusion_msgs__msg__RadarPoint
{
  float x;
  float y;
  float z;
  float intensity;
  float range;
  float azimuth;
  float elevation;
} fusion_msgs__msg__RadarPoint;

// Struct for a sequence of fusion_msgs__msg__RadarPoint.
typedef struct fusion_msgs__msg__RadarPoint__Sequence
{
  fusion_msgs__msg__RadarPoint * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} fusion_msgs__msg__RadarPoint__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // FUSION_MSGS__MSG__DETAIL__RADAR_POINT__STRUCT_H_
