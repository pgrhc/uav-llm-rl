// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from fusion_msgs:msg/RadarPoints.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "fusion_msgs/msg/detail/radar_points__rosidl_typesupport_introspection_c.h"
#include "fusion_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "fusion_msgs/msg/detail/radar_points__functions.h"
#include "fusion_msgs/msg/detail/radar_points__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `points`
#include "fusion_msgs/msg/radar_point.h"
// Member `points`
#include "fusion_msgs/msg/detail/radar_point__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__RadarPoints_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  fusion_msgs__msg__RadarPoints__init(message_memory);
}

void fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__RadarPoints_fini_function(void * message_memory)
{
  fusion_msgs__msg__RadarPoints__fini(message_memory);
}

size_t fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__size_function__RadarPoints__points(
  const void * untyped_member)
{
  const fusion_msgs__msg__RadarPoint__Sequence * member =
    (const fusion_msgs__msg__RadarPoint__Sequence *)(untyped_member);
  return member->size;
}

const void * fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__get_const_function__RadarPoints__points(
  const void * untyped_member, size_t index)
{
  const fusion_msgs__msg__RadarPoint__Sequence * member =
    (const fusion_msgs__msg__RadarPoint__Sequence *)(untyped_member);
  return &member->data[index];
}

void * fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__get_function__RadarPoints__points(
  void * untyped_member, size_t index)
{
  fusion_msgs__msg__RadarPoint__Sequence * member =
    (fusion_msgs__msg__RadarPoint__Sequence *)(untyped_member);
  return &member->data[index];
}

void fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__fetch_function__RadarPoints__points(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const fusion_msgs__msg__RadarPoint * item =
    ((const fusion_msgs__msg__RadarPoint *)
    fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__get_const_function__RadarPoints__points(untyped_member, index));
  fusion_msgs__msg__RadarPoint * value =
    (fusion_msgs__msg__RadarPoint *)(untyped_value);
  *value = *item;
}

void fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__assign_function__RadarPoints__points(
  void * untyped_member, size_t index, const void * untyped_value)
{
  fusion_msgs__msg__RadarPoint * item =
    ((fusion_msgs__msg__RadarPoint *)
    fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__get_function__RadarPoints__points(untyped_member, index));
  const fusion_msgs__msg__RadarPoint * value =
    (const fusion_msgs__msg__RadarPoint *)(untyped_value);
  *item = *value;
}

bool fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__resize_function__RadarPoints__points(
  void * untyped_member, size_t size)
{
  fusion_msgs__msg__RadarPoint__Sequence * member =
    (fusion_msgs__msg__RadarPoint__Sequence *)(untyped_member);
  fusion_msgs__msg__RadarPoint__Sequence__fini(member);
  return fusion_msgs__msg__RadarPoint__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__RadarPoints_message_member_array[2] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fusion_msgs__msg__RadarPoints, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "points",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fusion_msgs__msg__RadarPoints, points),  // bytes offset in struct
    NULL,  // default value
    fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__size_function__RadarPoints__points,  // size() function pointer
    fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__get_const_function__RadarPoints__points,  // get_const(index) function pointer
    fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__get_function__RadarPoints__points,  // get(index) function pointer
    fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__fetch_function__RadarPoints__points,  // fetch(index, &value) function pointer
    fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__assign_function__RadarPoints__points,  // assign(index, value) function pointer
    fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__resize_function__RadarPoints__points  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__RadarPoints_message_members = {
  "fusion_msgs__msg",  // message namespace
  "RadarPoints",  // message name
  2,  // number of fields
  sizeof(fusion_msgs__msg__RadarPoints),
  fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__RadarPoints_message_member_array,  // message members
  fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__RadarPoints_init_function,  // function to initialize message memory (memory has to be allocated)
  fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__RadarPoints_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__RadarPoints_message_type_support_handle = {
  0,
  &fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__RadarPoints_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_fusion_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, fusion_msgs, msg, RadarPoints)() {
  fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__RadarPoints_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__RadarPoints_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, fusion_msgs, msg, RadarPoint)();
  if (!fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__RadarPoints_message_type_support_handle.typesupport_identifier) {
    fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__RadarPoints_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &fusion_msgs__msg__RadarPoints__rosidl_typesupport_introspection_c__RadarPoints_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
