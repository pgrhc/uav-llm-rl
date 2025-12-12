// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from fusion_msgs:msg/RadarPoints.idl
// generated code does not contain a copyright notice
#include "fusion_msgs/msg/detail/radar_points__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `points`
#include "fusion_msgs/msg/detail/radar_point__functions.h"

bool
fusion_msgs__msg__RadarPoints__init(fusion_msgs__msg__RadarPoints * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    fusion_msgs__msg__RadarPoints__fini(msg);
    return false;
  }
  // points
  if (!fusion_msgs__msg__RadarPoint__Sequence__init(&msg->points, 0)) {
    fusion_msgs__msg__RadarPoints__fini(msg);
    return false;
  }
  return true;
}

void
fusion_msgs__msg__RadarPoints__fini(fusion_msgs__msg__RadarPoints * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // points
  fusion_msgs__msg__RadarPoint__Sequence__fini(&msg->points);
}

bool
fusion_msgs__msg__RadarPoints__are_equal(const fusion_msgs__msg__RadarPoints * lhs, const fusion_msgs__msg__RadarPoints * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // points
  if (!fusion_msgs__msg__RadarPoint__Sequence__are_equal(
      &(lhs->points), &(rhs->points)))
  {
    return false;
  }
  return true;
}

bool
fusion_msgs__msg__RadarPoints__copy(
  const fusion_msgs__msg__RadarPoints * input,
  fusion_msgs__msg__RadarPoints * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // points
  if (!fusion_msgs__msg__RadarPoint__Sequence__copy(
      &(input->points), &(output->points)))
  {
    return false;
  }
  return true;
}

fusion_msgs__msg__RadarPoints *
fusion_msgs__msg__RadarPoints__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fusion_msgs__msg__RadarPoints * msg = (fusion_msgs__msg__RadarPoints *)allocator.allocate(sizeof(fusion_msgs__msg__RadarPoints), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(fusion_msgs__msg__RadarPoints));
  bool success = fusion_msgs__msg__RadarPoints__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
fusion_msgs__msg__RadarPoints__destroy(fusion_msgs__msg__RadarPoints * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    fusion_msgs__msg__RadarPoints__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
fusion_msgs__msg__RadarPoints__Sequence__init(fusion_msgs__msg__RadarPoints__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fusion_msgs__msg__RadarPoints * data = NULL;

  if (size) {
    data = (fusion_msgs__msg__RadarPoints *)allocator.zero_allocate(size, sizeof(fusion_msgs__msg__RadarPoints), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = fusion_msgs__msg__RadarPoints__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        fusion_msgs__msg__RadarPoints__fini(&data[i - 1]);
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
fusion_msgs__msg__RadarPoints__Sequence__fini(fusion_msgs__msg__RadarPoints__Sequence * array)
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
      fusion_msgs__msg__RadarPoints__fini(&array->data[i]);
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

fusion_msgs__msg__RadarPoints__Sequence *
fusion_msgs__msg__RadarPoints__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fusion_msgs__msg__RadarPoints__Sequence * array = (fusion_msgs__msg__RadarPoints__Sequence *)allocator.allocate(sizeof(fusion_msgs__msg__RadarPoints__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = fusion_msgs__msg__RadarPoints__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
fusion_msgs__msg__RadarPoints__Sequence__destroy(fusion_msgs__msg__RadarPoints__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    fusion_msgs__msg__RadarPoints__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
fusion_msgs__msg__RadarPoints__Sequence__are_equal(const fusion_msgs__msg__RadarPoints__Sequence * lhs, const fusion_msgs__msg__RadarPoints__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!fusion_msgs__msg__RadarPoints__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
fusion_msgs__msg__RadarPoints__Sequence__copy(
  const fusion_msgs__msg__RadarPoints__Sequence * input,
  fusion_msgs__msg__RadarPoints__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(fusion_msgs__msg__RadarPoints);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    fusion_msgs__msg__RadarPoints * data =
      (fusion_msgs__msg__RadarPoints *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!fusion_msgs__msg__RadarPoints__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          fusion_msgs__msg__RadarPoints__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!fusion_msgs__msg__RadarPoints__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
