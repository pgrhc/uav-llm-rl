// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from fusion_msgs:msg/FusionStamped.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "fusion_msgs/msg/detail/fusion_stamped__rosidl_typesupport_introspection_c.h"
#include "fusion_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "fusion_msgs/msg/detail/fusion_stamped__functions.h"
#include "fusion_msgs/msg/detail/fusion_stamped__struct.h"


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/time.h"
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__rosidl_typesupport_introspection_c.h"
// Member `image`
#include "sensor_msgs/msg/image.h"
// Member `image`
#include "sensor_msgs/msg/detail/image__rosidl_typesupport_introspection_c.h"
// Member `lidar`
#include "sensor_msgs/msg/point_cloud2.h"
// Member `lidar`
#include "sensor_msgs/msg/detail/point_cloud2__rosidl_typesupport_introspection_c.h"
// Member `radar`
#include "fusion_msgs/msg/radar_points.h"
// Member `radar`
#include "fusion_msgs/msg/detail/radar_points__rosidl_typesupport_introspection_c.h"
// Member `odom`
#include "nav_msgs/msg/odometry.h"
// Member `odom`
#include "nav_msgs/msg/detail/odometry__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  fusion_msgs__msg__FusionStamped__init(message_memory);
}

void fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_fini_function(void * message_memory)
{
  fusion_msgs__msg__FusionStamped__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_message_member_array[5] = {
  {
    "stamp",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fusion_msgs__msg__FusionStamped, stamp),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "image",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fusion_msgs__msg__FusionStamped, image),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "lidar",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fusion_msgs__msg__FusionStamped, lidar),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "radar",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fusion_msgs__msg__FusionStamped, radar),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "odom",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fusion_msgs__msg__FusionStamped, odom),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_message_members = {
  "fusion_msgs__msg",  // message namespace
  "FusionStamped",  // message name
  5,  // number of fields
  sizeof(fusion_msgs__msg__FusionStamped),
  fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_message_member_array,  // message members
  fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_init_function,  // function to initialize message memory (memory has to be allocated)
  fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_message_type_support_handle = {
  0,
  &fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_fusion_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, fusion_msgs, msg, FusionStamped)() {
  fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, sensor_msgs, msg, Image)();
  fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, sensor_msgs, msg, PointCloud2)();
  fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, fusion_msgs, msg, RadarPoints)();
  fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_message_member_array[4].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, nav_msgs, msg, Odometry)();
  if (!fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_message_type_support_handle.typesupport_identifier) {
    fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &fusion_msgs__msg__FusionStamped__rosidl_typesupport_introspection_c__FusionStamped_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
