# Understanding shim.py

## The three bounding-box formats

### BoundingBoxXYXY

    x1
    y1
    x2
    y2

Meaning:

    (x1,y1) ┌───────────┐
            │           │
            │   bird    │
            │           │
            └───────────┘ (x2,y2)

---
### BoundingBoxXYWH

This contains:

    x
    y
    width
    height

where $x,y$ are still the top-left corner.

---
### BoundingBoxCXCYAH

This one is the Kalman representation:

    center_x
    center_y
    aspect_ratio
    height

So:

$$ [c_x,c_y,a,h] $$

where:

$$ a=\frac{w}{h} $$

This is exactly the measurement format used by the Kalman filter.

    CSV
    [x,y,w,h]
    ↓
    XYWH
    ↓
    run_tracker
    ↓
    XYXY
    [x1,y1,x2,y2]
    ↓
    Kalman update
    ↓
    CXCYAH
    [cx,cy,a,h]

---
---
## Helpers and conversions
### `xyxy_center()`

This helper computes:

$$ c_x=\frac{x_1+x_2}{2} $$ $$ c_y=\frac{y_1+y_2}{2} $$

---
### `xyxy_to_cxcyah()`

This is the conversion used before Kalman updates.

It computes:

    center_x, center_y = xyxy_center(...)
    aspect_ratio = width / height

and returns:

$$ [c_x,c_y,a,h] $$

---

### `xyxy_to_xywh()`

This conversion does:

$$ w=x_2-x_1 $$ $$ h=y_2-y_1 $$

and returns:

$$ [x_1,y_1,w,h] $$

This is used in distance_matching.py when the detection boxes are converted to the same TLWH format as the predicted Kalman boxes.

---

### `xywh_to_xyxy()`

The inverse:

$$ x_2=x+w $$ $$ y_2=y+h $$

This is used by run_tracker.py when raw CSV boxes become tracker detections.

---

### `distance_between_two_bounding_boxes()`

This function calculates the distance between box centers:

    (x1,y1), (x2,y2) = xyxy_center(...)
    return sqrt((x1-x2)^2 + (y1-y2)^2)

Important:

This is not the same distance used in distance_matching.py.

There:

    top-left corners are compared.

Here:

    centers are compared.

This helper is used in the later track-concatenation/post-processing logic, not the Stage B matching cost we studied.

---

## important Container

### `DetectionNoCrop`

```py
@dataclass
class DetectionNoCrop:
    bounding_box
    frame_number
    frame_timestamp
    feature
    ...
```

`bounding_box`

Where the detector saw the bird.

`frame_number`

Which frame this belongs to.

`feature`

The feature used by the Stage A nearest-neighbor matcher.

And in this standalone runner:

$$ feature=(x,y,\text{area}) $$