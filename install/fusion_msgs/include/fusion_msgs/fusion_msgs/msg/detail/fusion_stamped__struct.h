// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from fusion_msgs:msg/FusionStamped.idl
// generated code does not contain a copyright notice

#ifndef FUSION_MSGS__MSG__DETAIL__FUSION_STAMPED__STRUCT_H_
#define FUSION_MSGS__MSG__DETAIL__FUSION_STAMPED__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"
// Member 'image'
#include "sensor_msgs/msg/detail/image__struct.h"
// Member 'lidar'
#include "sensor_msgs/msg/detail/point_cloud2__struct.h"
// Member 'radar'
#include "fusion_msgs/msg/detail/radar_points__struct.h"
// Member 'odom'
#include "nav_msgs/msg/detail/odometry__struct.h"

/// Struct defined in msg/FusionStamped in the package fusion_msgs.
typedef struct fusion_msgs__msg__FusionStamped
{
  builtin_interfaces__msg__Time stamp;
  sensor_msgs__msg__Image image;
  sensor_msgs__msg__PointCloud2 lidar;
  fusion_msgs__msg__RadarPoints radar;
  nav_msgs__msg__Odometry odom;
} fusion_msgs__msg__FusionStamped;

// Struct for a sequence of fusion_msgs__msg__FusionStamped.
typedef struct fusion_msgs__msg__FusionStamped__Sequence
{
  fusion_msgs__msg__FusionStamped * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} fusion_msgs__msg__FusionStamped__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // FUSION_MSGS__MSG__DETAIL__FUSION_STAMPED__STRUCT_H_
