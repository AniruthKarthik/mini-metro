package engine

import "math"

type Pos struct {
	X float64
	Y float64
}

type RiverSegment struct {
	From  Pos
	To    Pos
	Width float64 // thickness of the river channel
}

type WaterPolygon struct {
	Vertices []Pos
}

// distance returns the Euclidean distance between two positions.
func distance(a, b Pos) float64 {
	dx := a.X - b.X
	dy := a.Y - b.Y
	return math.Sqrt(dx*dx + dy*dy)
}

// distToSegment calculates the minimum distance from point p to segment ab.
func distToSegment(p, a, b Pos) float64 {
	l2 := (b.X-a.X)*(b.X-a.X) + (b.Y-a.Y)*(b.Y-a.Y)
	if l2 == 0 {
		return distance(p, a)
	}
	t := ((p.X-a.X)*(b.X-a.X) + (p.Y-a.Y)*(b.Y-a.Y)) / l2
	if t < 0 {
		return distance(p, a)
	}
	if t > 1 {
		return distance(p, b)
	}
	proj := Pos{
		X: a.X + t*(b.X-a.X),
		Y: a.Y + t*(b.Y-a.Y),
	}
	return distance(p, proj)
}

// ccw returns the cross product orientation of points (a, b, c).
func ccw(a, b, c Pos) float64 {
	return (c.Y-a.Y)*(b.X-a.X) - (b.Y-a.Y)*(c.X-a.X)
}

// SegmentsIntersect reports whether line segment (a,b) intersects line segment (c,d).
func SegmentsIntersect(a, b, c, d Pos) bool {
	// Quick bounding box check
	if math.Min(a.X, b.X) > math.Max(c.X, d.X) || math.Max(a.X, b.X) < math.Min(c.X, d.X) ||
		math.Min(a.Y, b.Y) > math.Max(c.Y, d.Y) || math.Max(a.Y, b.Y) < math.Min(c.Y, d.Y) {
		return false
	}
	cp1 := ccw(a, b, c)
	cp2 := ccw(a, b, d)
	cp3 := ccw(c, d, a)
	cp4 := ccw(c, d, b)

	return ((cp1 > 0 && cp2 < 0) || (cp1 < 0 && cp2 > 0)) &&
		((cp3 > 0 && cp4 < 0) || (cp3 < 0 && cp4 > 0))
}

// PointInPolygon reports whether point p lies inside polygon poly using ray-casting.
func PointInPolygon(p Pos, poly []Pos) bool {
	if len(poly) < 3 {
		return false
	}
	inside := false
	j := len(poly) - 1
	for i := 0; i < len(poly); i++ {
		if (poly[i].Y > p.Y) != (poly[j].Y > p.Y) &&
			(p.X < (poly[j].X-poly[i].X)*(p.Y-poly[i].Y)/(poly[j].Y-poly[i].Y)+poly[i].X) {
			inside = !inside
		}
		j = i
	}
	return inside
}

// SegmentCrossesThickRiver reports whether segment (p1, p2) intersects or enters a thick river segment.
func SegmentCrossesThickRiver(p1, p2 Pos, r RiverSegment) bool {
	if SegmentsIntersect(p1, p2, r.From, r.To) {
		return true
	}
	halfWidth := r.Width / 2.0
	if halfWidth <= 0 {
		return false
	}
	if distToSegment(p1, r.From, r.To) <= halfWidth || distToSegment(p2, r.From, r.To) <= halfWidth {
		return true
	}
	return false
}

// SegmentCrossesWaterPolygon reports whether segment (p1, p2) intersects any edge or endpoint lies inside poly.
func SegmentCrossesWaterPolygon(p1, p2 Pos, poly WaterPolygon) bool {
	n := len(poly.Vertices)
	if n < 3 {
		return false
	}
	if PointInPolygon(p1, poly.Vertices) || PointInPolygon(p2, poly.Vertices) {
		return true
	}
	for i := 0; i < n; i++ {
		next := (i + 1) % n
		if SegmentsIntersect(p1, p2, poly.Vertices[i], poly.Vertices[next]) {
			return true
		}
	}
	return false
}

// CrossesWater reports whether line segment (p1, p2) intersects any thick river segment or 2D water polygon.
func CrossesWater(p1, p2 Pos, rivers []RiverSegment, polygons []WaterPolygon) bool {
	for _, r := range rivers {
		if SegmentCrossesThickRiver(p1, p2, r) {
			return true
		}
	}
	for _, poly := range polygons {
		if SegmentCrossesWaterPolygon(p1, p2, poly) {
			return true
		}
	}
	return false
}

func rewardInterval() uint64 {
	return 500
}


