"""
Unit tests for fwhm_arcmin_to_sigma_rad function

Tests the FWHM to sigma conversion function that computes:
σ = (fwhm_arcmin * π / 180 / 60) / sqrt(8 * log(2))

This converts a Gaussian beam's Full-Width at Half-Maximum from arcminutes
to its standard deviation in radians for use in CMB beam window calculations.
"""

@testset "fwhm_arcmin_to_sigma_rad() Unit Tests" begin

    @testset "Basic Functionality" begin
        # Test with typical CMB beam size
        fwhm_test = 5.0  # 5 arcminutes

        result = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm_test)

        # Basic output tests
        @test result isa Real
        @test result > 0.0  # Should be positive for positive FWHM
        @test isfinite(result)
        @test typeof(result) <: AbstractFloat

        # Should be in radians (small number for arcminute inputs)
        @test result < 1.0  # Less than 1 radian for typical beam sizes
        @test result > 1e-6  # But not too small (should be microradians to milliradians)
    end

    @testset "Mathematical Formula Verification" begin
        # Test exact mathematical formula: (fwhm * π/180/60) / sqrt(8*log(2))

        fwhm_values = [1.0, 5.0, 10.0, 30.0]

        for fwhm in fwhm_values
            result = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm)

            # Calculate expected value using the exact formula
            fwhm_radians = fwhm * π / 180 / 60  # Convert arcmin to radians
            expected = fwhm_radians / sqrt(8 * log(2))  # Convert FWHM to sigma

            @test result ≈ expected

            # Alternative calculation check
            # For Gaussian: FWHM = 2 * sqrt(2 * ln(2)) * σ ≈ 2.35482 * σ
            conversion_factor = 2 * sqrt(2 * log(2))
            expected_alt = fwhm_radians / conversion_factor

            @test result ≈ expected_alt
        end

        # Test specific known conversion
        fwhm_1arcmin = 1.0
        result_1arcmin = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm_1arcmin)

        # 1 arcminute = π/180/60 radians ≈ 4.848e-6 radians
        # σ = FWHM_rad / 2.35482 ≈ 2.06e-6 radians
        expected_1arcmin = (π / 180 / 60) / sqrt(8 * log(2))
        @test result_1arcmin ≈ expected_1arcmin
    end

    @testset "Unit Conversion Properties" begin
        # Test unit conversion properties

        # Linearity: σ(k*FWHM) = k*σ(FWHM)
        fwhm_base = 5.0
        scale = 3.0

        result_base = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm_base)
        result_scaled = CMBForegrounds.fwhm_arcmin_to_sigma_rad(scale * fwhm_base)

        @test result_scaled ≈ scale * result_base

        # Test ratio preservation
        fwhm1, fwhm2 = 4.0, 12.0
        sigma1 = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm1)
        sigma2 = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm2)

        @test sigma2 / sigma1 ≈ fwhm2 / fwhm1  # Ratios should be preserved

        # Zero input gives zero output
        result_zero = CMBForegrounds.fwhm_arcmin_to_sigma_rad(0.0)
        @test result_zero == 0.0
    end

    @testset "Gaussian Beam Relationship" begin
        # Test relationship with Gaussian beam parameters

        fwhm_test = 10.0  # arcminutes
        sigma_result = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm_test)

        # For a Gaussian beam, FWHM = 2*sqrt(2*ln(2))*σ ≈ 2.35482*σ
        gaussian_factor = 2 * sqrt(2 * log(2))

        # Convert back to FWHM in radians
        fwhm_rad_back = gaussian_factor * sigma_result
        fwhm_arcmin_back = fwhm_rad_back * 180 * 60 / π

        @test fwhm_arcmin_back ≈ fwhm_test  # Should recover original FWHM

        # Test the Gaussian factor directly
        @test gaussian_factor ≈ 2.3548200450309493  # Known value

        # Verify relationship: σ = FWHM_rad / gaussian_factor
        fwhm_rad = fwhm_test * π / 180 / 60
        expected_sigma = fwhm_rad / gaussian_factor
        @test sigma_result ≈ expected_sigma
    end

    @testset "Physical Realism - CMB Instruments" begin
        # Test with realistic CMB instrument beam sizes

        # Typical CMB experiment FWHM values (arcminutes)
        realistic_beams = [
            0.5,    # Very high resolution ground-based
            1.0,    # High resolution ground-based
            1.5,    # ACTPol, SPT-3G
            5.0,    # Planck HFI high freq
            10.0,   # Planck HFI mid freq
            15.0,   # Planck HFI low freq
            30.0,   # Planck LFI
            60.0,   # WMAP W-band
        ]

        for fwhm in realistic_beams
            sigma = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm)

            # Physical constraints
            @test sigma > 0.0
            @test isfinite(sigma)

            # Reasonable range for CMB beams (should be in microradians to milliradians)
            @test 1e-7 < sigma < 1e-2  # 0.1 μrad to 10 mrad

            # Should scale linearly with FWHM
            if fwhm > 1.0
                sigma_1arcmin = CMBForegrounds.fwhm_arcmin_to_sigma_rad(1.0)
                expected_ratio = fwhm / 1.0
                actual_ratio = sigma / sigma_1arcmin
                @test actual_ratio ≈ expected_ratio
            end
        end

        # Test very precise instruments (sub-arcminute)
        precise_fwhm = 0.1  # 6 arcseconds
        sigma_precise = CMBForegrounds.fwhm_arcmin_to_sigma_rad(precise_fwhm)
        @test sigma_precise > 0.0
        @test sigma_precise < 5e-5  # Small sigma for precise beam

        # Test low resolution (degree-scale)
        broad_fwhm = 120.0  # 2 degrees
        sigma_broad = CMBForegrounds.fwhm_arcmin_to_sigma_rad(broad_fwhm)
        @test sigma_broad > 1e-4  # Larger sigma for broad beam
        @test sigma_broad < 1.0    # But still reasonable
    end

    @testset "Type Stability and Promotion" begin
        # Test with different input types

        # Float64 input
        result_float = CMBForegrounds.fwhm_arcmin_to_sigma_rad(5.0)
        @test typeof(result_float) == Float64

        # Integer input
        result_int = CMBForegrounds.fwhm_arcmin_to_sigma_rad(5)
        @test typeof(result_int) <: AbstractFloat  # Should promote to float
        @test result_int ≈ result_float  # Should give same numerical result

        # Float32 input
        result_float32 = CMBForegrounds.fwhm_arcmin_to_sigma_rad(5.0f0)
        @test typeof(result_float32) <: AbstractFloat
        @test result_float32 ≈ result_float atol = 1e-6  # Close to Float64 result

        # BigFloat input
        result_big = CMBForegrounds.fwhm_arcmin_to_sigma_rad(big(5.0))
        @test typeof(result_big) == BigFloat
        @test Float64(result_big) ≈ result_float  # Same value in Float64

        # Rational input
        result_rational = CMBForegrounds.fwhm_arcmin_to_sigma_rad(5 // 1)
        @test typeof(result_rational) <: AbstractFloat
        @test result_rational ≈ result_float
    end

    @testset "Edge Cases and Boundary Conditions" begin
        # Test edge cases

        # Very small FWHM (sub-arcsecond precision)
        fwhm_tiny = 1e-6  # 1 micro-arcminute
        result_tiny = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm_tiny)
        @test result_tiny > 0.0
        @test result_tiny < 1e-9  # Very small sigma
        @test isfinite(result_tiny)

        # Very large FWHM (degree scale)
        fwhm_large = 1000.0  # 1000 arcminutes ≈ 16.67 degrees
        result_large = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm_large)
        @test result_large > 0.0
        @test result_large < π  # Less than π radians (180 degrees)
        @test isfinite(result_large)

        # Zero FWHM
        result_zero = CMBForegrounds.fwhm_arcmin_to_sigma_rad(0.0)
        @test result_zero == 0.0

        # Test ordering: larger FWHM gives larger sigma
        fwhm_values = [0.1, 1.0, 10.0, 100.0]
        sigma_values = [CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm) for fwhm in fwhm_values]
        @test issorted(sigma_values)  # Should be monotonically increasing
    end

    @testset "Numerical Precision" begin
        # Test numerical precision and stability

        # High precision input
        fwhm_precise = π  # Use π for precision test
        result_precise = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm_precise)

        # Calculate expected with high precision
        expected_precise = (fwhm_precise * π / 180 / 60) / sqrt(8 * log(2))
        @test result_precise ≈ expected_precise rtol = 1e-14

        # Test with mathematical constants
        fwhm_e = ℯ
        result_e = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm_e)
        expected_e = (ℯ * π / 180 / 60) / sqrt(8 * log(2))
        @test result_e ≈ expected_e rtol = 1e-14

        # Test numerical stability near zero
        small_values = [1e-10, 1e-8, 1e-6, 1e-4]
        for fwhm_small in small_values
            result_small = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm_small)
            expected_small = (fwhm_small * π / 180 / 60) / sqrt(8 * log(2))
            @test result_small ≈ expected_small rtol = 1e-12
            @test result_small > 0.0
        end

        # Test with values that might cause overflow/underflow
        extreme_values = [1e-20, 1e20]  # Very small and very large
        for fwhm_extreme in extreme_values
            result_extreme = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm_extreme)
            @test isfinite(result_extreme)
            @test result_extreme >= 0.0
        end
    end

    @testset "Conversion Constants and Factors" begin
        # Test the conversion constants used in the function

        # Test arcminute to radian conversion
        arcmin_to_rad = π / 180 / 60  # Conversion factor
        @test arcmin_to_rad ≈ 2.908882086657216e-4  # Known value: π/180/60

        # Test Gaussian FWHM to sigma conversion
        gaussian_factor = sqrt(8 * log(2))  # FWHM = gaussian_factor * σ
        @test gaussian_factor ≈ 2.3548200450309493  # Known value

        # Test combined conversion factor
        combined_factor = arcmin_to_rad / gaussian_factor
        fwhm_test = 1.0
        result_direct = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm_test)
        result_factor = fwhm_test * combined_factor
        @test result_direct ≈ result_factor

        # Verify the mathematical relationship
        @test combined_factor ≈ π / (180 * 60 * sqrt(8 * log(2)))
    end

    @testset "Comparison with Literature Values" begin
        # Test against known values from CMB literature

        # Planck beam sizes (approximate - corrected values)
        planck_beams = [
            (30.0, 3.7e-3),    # 30 arcmin FWHM → σ ≈ 3.7e-3 rad
            (44.0, 5.4e-3),    # 44 arcmin FWHM → σ ≈ 5.4e-3 rad
            (70.0, 8.6e-3),    # 70 arcmin FWHM → σ ≈ 8.6e-3 rad
        ]

        # Note: These are approximate values for testing order of magnitude
        for (fwhm_arcmin, expected_order) in planck_beams
            sigma = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm_arcmin)

            # Test order of magnitude (within factor of 3)
            @test sigma > expected_order / 3
            @test sigma < expected_order * 3
            @test sigma > 0.0
        end

        # Reference calculation: 1 arcminute FWHM
        # 1 arcmin = π/180/60 rad ≈ 2.909e-4 rad
        # σ = 2.909e-4 / 2.3548 ≈ 1.235e-4 rad
        sigma_1arcmin = CMBForegrounds.fwhm_arcmin_to_sigma_rad(1.0)
        @test sigma_1arcmin ≈ 1.235e-4 rtol = 0.01
    end

    @testset "Function Properties" begin
        # Test mathematical properties of the function

        # Linearity: f(ax) = a*f(x) for a > 0
        a_values = [0.5, 2.0, 10.0]
        x = 5.0

        for a in a_values
            result_ax = CMBForegrounds.fwhm_arcmin_to_sigma_rad(a * x)
            result_a_fx = a * CMBForegrounds.fwhm_arcmin_to_sigma_rad(x)
            @test result_ax ≈ result_a_fx
        end

        # Monotonicity: f(x₁) < f(x₂) if x₁ < x₂
        x_values = [0.1, 1.0, 5.0, 20.0, 100.0]
        sigma_values = [CMBForegrounds.fwhm_arcmin_to_sigma_rad(x) for x in x_values]

        for i in 2:length(sigma_values)
            @test sigma_values[i] > sigma_values[i-1]
        end

        # Positive homogeneity: f(tx) = t*f(x) for t > 0
        t = 7.3
        x = 2.5
        result_tx = CMBForegrounds.fwhm_arcmin_to_sigma_rad(t * x)
        result_t_fx = t * CMBForegrounds.fwhm_arcmin_to_sigma_rad(x)
        @test result_tx ≈ result_t_fx
    end

    @testset "Inline Function Performance" begin
        # Test that @inline annotation doesn't affect correctness

        # The function should work the same despite being @inline
        test_values = [0.1, 1.0, 5.0, 10.0, 50.0]

        for fwhm in test_values
            result = CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm)
            expected = (fwhm * π / 180 / 60) / sqrt(8 * log(2))
            @test result ≈ expected
        end

        # Test vectorized-like usage (calling multiple times)
        results = [CMBForegrounds.fwhm_arcmin_to_sigma_rad(fwhm) for fwhm in test_values]
        expected_results = [(fwhm * π / 180 / 60) / sqrt(8 * log(2)) for fwhm in test_values]

        for i in 1:length(results)
            @test results[i] ≈ expected_results[i]
        end
    end

end
