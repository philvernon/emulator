package emulator

import (
	"testing"

	"github.com/goccy/go-yaml"
	"github.com/philvernon/emulator/anomaly"
	"github.com/stretchr/testify/assert"
)

// continuousPositiveSpike returns a spike anomaly that fires every time step with a fixed
// positive magnitude.
func continuousPositiveSpike(name string, magnitude float64) (anomaly.AnomalyInterface, error) {
	return anomaly.NewSpikeAnomaly(anomaly.SpikeParams{
		Name:        name,
		Probability: 1.0,
		Magnitude:   magnitude,
		SpikeSign:   1.0,
	})
}

func assertAllClose(t *testing.T, got []float64, want float64) {
	t.Helper()
	assert.Len(t, got, len(got))
	for i, v := range got {
		assert.InDelta(t, want, v, 1e-9, "point %d", i)
	}
}

func TestDtsEmulation_UniformBaseline(t *testing.T) {
	d, err := NewDtsEmulation(DtsEmulation{
		PointCount:         3,
		SpatialResolutionM: 1.0,
		BaseTemperatures:   []float64{18.4},
	})
	assert.NoError(t, err)

	emu := NewEmulator(100, 50)
	emu.SetRandomSeed(1)
	emu.D = d
	emu.Step()

	assertAllClose(t, d.T, 18.4)
}

func TestDtsEmulation_SpatialBaseline(t *testing.T) {
	d, err := NewDtsEmulation(DtsEmulation{
		PointCount:         3,
		SpatialResolutionM: 1.0,
		BaseTemperatures:   []float64{18.4, 18.5, 18.6},
	})
	assert.NoError(t, err)

	emu := NewEmulator(100, 50)
	emu.SetRandomSeed(1)
	emu.D = d
	emu.Step()

	want := []float64{18.4, 18.5, 18.6}
	assert.Equal(t, want, d.T)
	assert.Equal(t, []float64{0, 1, 2}, d.Distances())
}

func TestDtsEmulation_DefaultBaseline(t *testing.T) {
	d, err := NewDtsEmulation(DtsEmulation{
		PointCount:         2,
		SpatialResolutionM: 10.0,
	})
	assert.NoError(t, err)

	emu := NewEmulator(100, 50)
	emu.SetRandomSeed(1)
	emu.D = d
	emu.Step()

	assertAllClose(t, d.T, defaultDtsBaselineCelsius)
}

func TestDtsEmulation_Noise(t *testing.T) {
	d, err := NewDtsEmulation(DtsEmulation{
		PointCount:         100,
		SpatialResolutionM: 1.0,
		BaseTemperatures:   []float64{20.0},
		NoiseMag:           0.01,
	})
	assert.NoError(t, err)

	emu := NewEmulator(100, 50)
	emu.SetRandomSeed(1)
	emu.D = d

	sum := 0.0
	for i := 0; i < 1000; i++ {
		emu.Step()
	}
	for _, v := range d.T {
		sum += v
	}
	// Mean of the final profile should sit near the baseline
	assert.InDelta(t, 20.0, sum/float64(len(d.T)), 0.1)
}

func TestDtsEmulation_GlobalAnomaly(t *testing.T) {
	spike, err := continuousPositiveSpike("global", 5.0)
	assert.NoError(t, err)

	d, err := NewDtsEmulation(DtsEmulation{
		PointCount:         4,
		SpatialResolutionM: 1.0,
		BaseTemperatures:   []float64{18.0},
	})
	assert.NoError(t, err)
	err = d.AddAnomaly(0, 0, spike)
	assert.NoError(t, err)

	emu := NewEmulator(100, 50)
	emu.SetRandomSeed(1)
	emu.D = d
	emu.Step()

	assertAllClose(t, d.T, 23.0)
}

func TestDtsEmulation_SpatialWindow(t *testing.T) {
	spike, err := continuousPositiveSpike("hotspot", 5.0)
	assert.NoError(t, err)

	// Points at 0, 10, ..., 90 m; window [20, 50) covers points 20, 30, 40
	d, err := NewDtsEmulation(DtsEmulation{
		PointCount:         10,
		SpatialResolutionM: 10.0,
		BaseTemperatures:   []float64{18.0},
	})
	assert.NoError(t, err)
	err = d.AddAnomaly(20, 30, spike)
	assert.NoError(t, err)

	emu := NewEmulator(100, 50)
	emu.SetRandomSeed(1)
	emu.D = d
	emu.Step()

	want := []float64{18, 18, 23, 23, 23, 18, 18, 18, 18, 18}
	for i, v := range d.T {
		assert.InDelta(t, want[i], v, 1e-9, "point %d", i)
	}
}

func TestDtsEmulation_MultipleWindows(t *testing.T) {
	hot, err := continuousPositiveSpike("hot", 3.0)
	assert.NoError(t, err)
	cold, err := continuousPositiveSpike("cold", 1.0)
	assert.NoError(t, err)

	// Points at 0, 5, ..., 25 m
	d, err := NewDtsEmulation(DtsEmulation{
		PointCount:         6,
		SpatialResolutionM: 5.0,
		BaseTemperatures:   []float64{20.0},
	})
	assert.NoError(t, err)
	err = d.AddAnomaly(0, 10, hot)   // points 0, 5
	err = d.AddAnomaly(10, 10, cold) // points 10, 15
	assert.NoError(t, err)

	emu := NewEmulator(100, 50)
	emu.SetRandomSeed(1)
	emu.D = d
	emu.Step()

	want := []float64{23, 23, 21, 21, 20, 20}
	for i, v := range d.T {
		assert.InDelta(t, want[i], v, 1e-9, "point %d", i)
	}
}

func TestDtsEmulation_Deterministic(t *testing.T) {
	build := func() *DtsEmulation {
		spike, err := continuousPositiveSpike("hot", 3.0)
		assert.NoError(t, err)
		d, err := NewDtsEmulation(DtsEmulation{
			PointCount:         6,
			SpatialResolutionM: 5.0,
			BaseTemperatures:   []float64{20.0, 20.1, 20.2, 20.3, 20.4, 20.5},
			NoiseMag:           0.01,
		})
		assert.NoError(t, err)
		assert.NoError(t, d.AddAnomaly(5, 10, spike))
		return d
	}

	a, b := build(), build()
	ea := NewEmulator(100, 50)
	ea.SetRandomSeed(42)
	ea.D = a
	eb := NewEmulator(100, 50)
	eb.SetRandomSeed(42)
	eb.D = b

	for i := 0; i < 50; i++ {
		ea.Step()
		eb.Step()
	}
	assert.Equal(t, a.T, b.T)
}

func TestDtsEmulation_DoesNotAffectTemperature(t *testing.T) {
	run := func(withDts bool) []float64 {
		emu := NewEmulator(100, 50)
		emu.SetRandomSeed(7)
		emu.T = &TemperatureEmulation{MeanTemperature: 30.0, NoiseMag: 0.01}
		if withDts {
			// Zero noise and no anomalies: D makes no PRNG draws, so the
			// shared stream feeding T must be untouched.
			d, err := NewDtsEmulation(DtsEmulation{
				PointCount:         4,
				SpatialResolutionM: 1.0,
				BaseTemperatures:   []float64{18.0},
			})
			assert.NoError(t, err)
			emu.D = d
		}
		out := make([]float64, 20)
		for i := range out {
			emu.Step()
			out[i] = emu.T.T
		}
		return out
	}
	assert.Equal(t, run(false), run(true))
}

func TestDtsEmulation_Validation(t *testing.T) {
	cases := []struct {
		name string
		cfg  DtsEmulation
	}{
		{"zero point count", DtsEmulation{PointCount: 0, SpatialResolutionM: 1}},
		{"zero resolution", DtsEmulation{PointCount: 3, SpatialResolutionM: 0}},
		{"base length mismatch", DtsEmulation{PointCount: 4, SpatialResolutionM: 1, BaseTemperatures: []float64{1, 2, 3}}},
		{"negative window start", DtsEmulation{
			PointCount: 4, SpatialResolutionM: 1,
			Anomaly: []DtsAnomalyGroup{{DistanceM: -1}},
		}},
		{"negative window width", DtsEmulation{
			PointCount: 4, SpatialResolutionM: 1,
			Anomaly: []DtsAnomalyGroup{{WidthM: -1}},
		}},
		{"window past fiber end", DtsEmulation{
			PointCount: 4, SpatialResolutionM: 1, // points at 0..3 m
			Anomaly: []DtsAnomalyGroup{{DistanceM: 2, WidthM: 2}},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewDtsEmulation(tc.cfg)
			assert.Error(t, err)
		})
	}

	// Valid configurations
	for _, cfg := range []DtsEmulation{
		{PointCount: 3, SpatialResolutionM: 1},
		{PointCount: 3, SpatialResolutionM: 1, BaseTemperatures: []float64{18}},
		{PointCount: 3, SpatialResolutionM: 1, BaseTemperatures: []float64{18, 19, 20}},
		{
			PointCount: 4, SpatialResolutionM: 1,
			Anomaly: []DtsAnomalyGroup{{DistanceM: 1.5, WidthM: 1.5}},
		},
	} {
		t.Run("valid", func(t *testing.T) {
			_, err := NewDtsEmulation(cfg)
			assert.NoError(t, err)
		})
	}
}

func TestDtsEmulation_UnmarshalYAML(t *testing.T) {
	yamlDoc := `
PointCount: 3
SpatialResolutionM: 1.0
BaseTemperatures: [18.4]
NoiseMag: 0.0
Anomaly:
  - DistanceM: 1
    WidthM: 1
    Anomalies:
      - Name: hotspot
        Type: spike
        Probability: 1.0
        Magnitude: 5
        Sign: 1.0
`
	var d DtsEmulation
	err := yaml.Unmarshal([]byte(yamlDoc), &d)
	assert.NoError(t, err)
	assert.Len(t, d.T, 3)

	emu := NewEmulator(100, 50)
	emu.SetRandomSeed(1)
	emu.D = &d
	emu.Step()

	// Window [1, 2) covers only point 1 (at 1 m)
	want := []float64{18.4, 23.4, 18.4}
	for i, v := range d.T {
		assert.InDelta(t, want[i], v, 1e-9, "point %d", i)
	}
}

func TestDtsEmulation_UnmarshalYAMLValidation(t *testing.T) {
	yamlDoc := `
PointCount: 3
SpatialResolutionM: 1.0
BaseTemperatures: [18.4, 18.5]
`
	var d DtsEmulation
	err := yaml.Unmarshal([]byte(yamlDoc), &d)
	assert.Error(t, err)
}
