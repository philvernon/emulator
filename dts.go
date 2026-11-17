package emulator

import (
	"errors"
	"fmt"
	"math/rand/v2"

	"github.com/philvernon/emulator/anomaly"
)

// defaultDtsBaselineCelsius is the baseline temperature used when BaseTemperatures is empty.
const defaultDtsBaselineCelsius = 20.0

// DtsAnomalyGroup is a set of anomalies applied to a spatial window of the fiber.
// DistanceM and WidthM are metres from the start of the fiber; WidthM of 0
// (the default) applies the group to the whole fiber.
type DtsAnomalyGroup struct {
	Anomalies anomaly.Container `yaml:"Anomalies"`
	DistanceM float64           `yaml:"DistanceM"`
	WidthM    float64           `yaml:"WidthM"`
}

// DtsEmulation emulates a DTS (distributed temperature sensing) profile: PointCount
// temperature samples spaced SpatialResolutionM apart along a fiber.
//
// BaseTemperatures may hold a single uniform baseline, PointCount values forming
// a spatial profile, or be empty (uses defaultDtsBaselineCelsius).
type DtsEmulation struct {
	PointCount         int               `yaml:"PointCount"`
	SpatialResolutionM float64           `yaml:"SpatialResolutionM"`
	BaseTemperatures   []float64         `yaml:"BaseTemperatures"`
	NoiseMag           float64           `yaml:"NoiseMag"`
	Anomaly            []DtsAnomalyGroup `yaml:"Anomaly"`

	T []float64 `yaml:"-"` // present value of each point
}

// dtsEmulationParams mirrors the config fields of DtsEmulation so UnmarshalYAML
// can decode without recursing and route through NewDtsEmulation's validation.
type dtsEmulationParams struct {
	PointCount         int               `yaml:"PointCount"`
	SpatialResolutionM float64           `yaml:"SpatialResolutionM"`
	BaseTemperatures   []float64         `yaml:"BaseTemperatures"`
	NoiseMag           float64           `yaml:"NoiseMag"`
	Anomaly            []DtsAnomalyGroup `yaml:"Anomaly"`
}

// UnmarshalYAML decodes a DtsEmulation from yaml, validating it via NewDtsEmulation.
func (d *DtsEmulation) UnmarshalYAML(unmarshal func(any) error) error {
	var params dtsEmulationParams
	if err := unmarshal(&params); err != nil {
		return err
	}
	newD, err := NewDtsEmulation(DtsEmulation{
		PointCount:         params.PointCount,
		SpatialResolutionM: params.SpatialResolutionM,
		BaseTemperatures:   params.BaseTemperatures,
		NoiseMag:           params.NoiseMag,
		Anomaly:            params.Anomaly,
	})
	if err != nil {
		return err
	}
	*d = *newD
	return nil
}

// NewDtsEmulation validates cfg and returns a ready-to-step DTS profile emulator.
func NewDtsEmulation(cfg DtsEmulation) (*DtsEmulation, error) {
	d := cfg
	if err := d.Validate(); err != nil {
		return nil, err
	}
	d.T = make([]float64, d.PointCount)
	return &d, nil
}

// Validate checks the configuration: positive PointCount and SpatialResolutionM,
// BaseTemperatures of length 0, 1, or PointCount, and anomaly windows that fit
// within the fiber.
func (d *DtsEmulation) Validate() error {
	if d.PointCount <= 0 {
		return errors.New("PointCount must be greater than 0")
	}
	if d.SpatialResolutionM <= 0 {
		return errors.New("SpatialResolutionM must be greater than 0")
	}
	if n := len(d.BaseTemperatures); n != 0 && n != 1 && n != d.PointCount {
		return fmt.Errorf("BaseTemperatures must contain 0, 1, or PointCount (%d) values, got %d", d.PointCount, n)
	}
	fiberEnd := float64(d.PointCount-1) * d.SpatialResolutionM
	for i := range d.Anomaly {
		g := &d.Anomaly[i]
		if g.DistanceM < 0 {
			return fmt.Errorf("anomaly group %d: DistanceM must be >= 0", i)
		}
		if g.WidthM < 0 {
			return fmt.Errorf("anomaly group %d: WidthM must be >= 0", i)
		}
		if g.WidthM > 0 && g.DistanceM+g.WidthM > fiberEnd {
			return fmt.Errorf("anomaly group %d: window extends past the end of the fiber (%.1f m)", i, fiberEnd)
		}
	}
	return nil
}

// baseAt returns the baseline temperature of point i.
func (d *DtsEmulation) baseAt(i int) float64 {
	switch len(d.BaseTemperatures) {
	case 0:
		return defaultDtsBaselineCelsius
	case 1:
		return d.BaseTemperatures[0]
	default:
		return d.BaseTemperatures[i]
	}
}

// DistanceM returns the distance in metres from the start of the fiber of point i.
func (d *DtsEmulation) DistanceM(i int) float64 {
	return float64(i) * d.SpatialResolutionM
}

// Distances returns the distance in metres from the start of the fiber of each point.
func (d *DtsEmulation) Distances() []float64 {
	distances := make([]float64, d.PointCount)
	for i := range distances {
		distances[i] = d.DistanceM(i)
	}
	return distances
}

// AddAnomaly adds an anomaly to the group applying to the window [startM, startM+widthM).
// widthM of 0 targets the whole fiber, creating a group if none exists yet.
func (d *DtsEmulation) AddAnomaly(startM, widthM float64, anom anomaly.AnomalyInterface) error {
	for i := range d.Anomaly {
		if d.Anomaly[i].DistanceM == startM && d.Anomaly[i].WidthM == widthM {
			return d.Anomaly[i].Anomalies.AddAnomaly(anom)
		}
	}
	d.Anomaly = append(d.Anomaly, DtsAnomalyGroup{
		Anomalies: anomaly.Container{anom},
		DistanceM: startM,
		WidthM:    widthM,
	})
	return nil
}

// Steps the DTS profile emulation forward by one time step. Each point is
// calculated as its baseline + Gaussian noise + the sum of the anomaly groups
// covering it. No random draws are made when NoiseMag is 0 and no anomaly
// groups are configured, so a quiet DTS profile leaves the shared PRNG stream
// for other emulators untouched.
func (d *DtsEmulation) stepDts(r *rand.Rand, Ts float64) {
	if d.T == nil {
		d.T = make([]float64, d.PointCount)
	}
	for i := range d.T {
		base := d.baseAt(i)
		d.T[i] = base
		if d.NoiseMag != 0 {
			d.T[i] += r.NormFloat64() * d.NoiseMag * base
		}
	}
	for i := range d.Anomaly {
		g := &d.Anomaly[i]
		delta := g.Anomalies.StepAll(r, Ts)
		if delta == 0 {
			continue
		}
		for p := 0; p < d.PointCount; p++ {
			dist := d.DistanceM(p)
			if dist < g.DistanceM {
				continue
			}
			if g.WidthM > 0 && dist >= g.DistanceM+g.WidthM {
				continue
			}
			d.T[p] += delta
		}
	}
}
