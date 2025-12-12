// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from fusion_msgs:msg/FusionStamped.idl
// generated code does not contain a copyright notice
#include "fusion_msgs/msg/detail/fusion_stamped__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__functions.h"
// Member `image`
#include "sensor_msgs/msg/detail/image__functions.h"
// Member `lidar`
#include "sensor_msgs/msg/detail/point_cloud2__functions.h"
// Member `radar`
#include "fusion_msgs/msg/detail/radar_points__functions.h"
// Member `odom`
#include "nav_msgs/msg/detail/odometry__functions.h"

bool
fusion_msgs__msg__FusionStamped__init(fusion_msgs__msg__FusionStamped * msg)
{
  if (!msg) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    fusion_msgs__msg__FusionStamped__fini(msg);
    return false;
  }
  // image
  if (!sensor_msgs__msg__Image__init(&msg->image)) {
    fusion_msgs__msg__FusionStamped__fini(msg);
    return false;
  }
  // lidar
  if (!sensor_msgs__msg__PointCloud2__init(&msg->lidar)) {
    fusion_msgs__msg__FusionStamped__fini(msg);
    return false;
  }
  // radar
  if (!fusion_msgs__msg__RadarPoints__init(&msg->radar)) {
    fusion_msgs__msg__FusionStamped__fini(msg);
    return false;
  }
  // odom
  if (!nav_msgs__msg__Odometry__init(&msg->odom)) {
    fusion_msgs__msg__FusionStamped__fini(msg);
    return false;
  }
  return true;
}

void
fusion_msgs__msg__FusionStamped__fini(fusion_msgs__msg__FusionStamped * msg)
{
  if (!msg) {
    return;
  }
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
  // image
  sensor_msgs__msg__Image__fini(&msg->image);
  // lidar
  sensor_msgs__msg__PointCloud2__fini(&msg->lidar);
  // radar
  fusion_msgs__msg__RadarPoints__fini(&msg->radar);
  // odom
  nav_msgs__msg__Odometry__fini(&msg->odom);
}

bool
fusion_msgs__msg__FusionStamped__are_equal(const fusion_msgs__msg__FusionStamped * lhs, const fusion_msgs__msg__FusionStamped * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->stamp), &(rhs->stamp)))
  {
    return false;
  }
  // image
  if (!sensor_msgs__msg__Image__are_equal(
      &(lhs->image), &(rhs->image)))
  {
    return false;
  }
  // lidar
  if (!sensor_msgs__msg__PointCloud2__are_equal(
      &(lhs->lidar), &(rhs->lidar)))
  {
    return false;
  }
  // radar
  if (!fusion_msgs__msg__RadarPoints__are_equal(
      &(lhs->radar), &(rhs->radar)))
  {
    return false;
  }
  // odom
  if (!nav_msgs__msg__Odometry__are_equal(
      &(lhs->odom), &(rhs->odom)))
  {
    return false;
  }
  return true;
}

bool
fusion_msgs__msg__FusionStamped__copy(
  const fusion_msgs__msg__FusionStamped * input,
  fusion_msgs__msg__FusionStamped * output)
{
  if (!input || !output) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  // image
  if (!sensor_msgs__msg__Image__copy(
      &(input->image), &(output->image)))
  {
    return false;
  }
  // lidar
  if (!sensor_msgs__msg__PointCloud2__copy(
      &(input->lidar), &(output->lidar)))
  {
    return false;
  }
  // radar
  if (!fusion_msgs__msg__RadarPoints__copy(
      &(input->radar), &(output->radar)))
  {
    return false;
  }
  // odom
  if (!nav_msgs__msg__Odometry__copy(
      &(input->odom), &(output->odom)))
  {
    return false;
  }
  return true;
}

fusion_msgs__msg__FusionStamped *
fusion_msgs__msg__FusionStamped__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fusion_msgs__msg__FusionStamped * msg = (fusion_msgs__msg__FusionStamped *)allocator.allocate(sizeof(fusion_msgs__msg__FusionStamped), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(fusion_msgs__msg__FusionStamped));
  bool success = fusion_msgs__msg__FusionStamped__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
fusion_msgs__msg__FusionStamped__destroy(fusion_msgs__msg__FusionStamped * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    fusion_msgs__msg__FusionStamped__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
fusion_msgs__msg__FusionStamped__Sequence__init(fusion_msgs__msg__FusionStamped__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fusion_msgs__msg__FusionStamped * data = NULL;

  if (size) {
    data = (fusion_msgs__msg__FusionStamped *)allocator.zero_allocate(size, sizeof(fusion_msgs__msg__FusionStamped), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = fusion_msgs__msg__FusionStamped__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        fusion_msgs__msg__FusionStamped__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
fusion_msgs__msg__FusionStamped__Sequence__fini(fusion_msgs__msg__FusionStamped__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      fusion_msgs__msg__FusionStamped__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

fusion_msgs__msg__FusionStamped__Sequence *
fusion_msgs__msg__FusionStamped__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fusion_msgs__msg__FusionStamped__Sequence * array = (fusion_msgs__msg__FusionStamped__Sequence *)allocator.allocate(sizeof(fusion_msgs__msg__FusionStamped__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = fusion_msgs__msg__FusionStamped__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
fusion_msgs__msg__FusionStamped__Sequence__destroy(fusion_msgs__msg__FusionStamped__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    fusion_msgs__msg__FusionStamped__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
fusion_msgs__msg__FusionStamped__Sequence__are_equal(const fusion_msgs__msg__FusionStamped__Sequence * lhs, const fusion_msgs__msg__FusionStamped__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!fusion_msgs__msg__FusionStamped__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
fusion_msgs__msg__FusionStamped__Sequence__copy(
  const fusion_msgs__msg__FusionStamped__Sequence * input,
  fusion_msgs__msg__FusionStamped__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(fusion_msgs__msg__FusionStamped);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    fusion_msgs__msg__FusionStamped * data =
      (fusion_msgs__msg__FusionStamped *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!fusion_msgs__msg__FusionStamped__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          fusion_msgs__msg__FusionStamped__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!fusion_msgs__msg__FusionStamped__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
