using GeometryBasics, LinearAlgebra, GLMakie
#using Makie: get_dim, surface_normals
using Images, ImageFiltering
using Chain
import NCDatasets as nc
import RData as rd
using Colors, ColorSchemes
using Statistics
using Dates
using YAXArrays, Zarr
using Interpolations
using ProgressBars
using Rasters
import Shapefile, Extents
import Downloads
using MRTools

# Earth Image
const earth_img = load(download("https://svs.gsfc.nasa.gov/vis/a000000/a002900/a002915/bluemarble-2048.png"));
#dem05deg=dem05()
dem05deg = nc.Dataset(joinpath(@__DIR__, "..", "data", "globalDEM0.5deg.nc"))[:unnamed][:]
##
GLMakie.activate!()

function grid2surfpoints(::Val{:sphere}, grd=fill(0.0, 180, 90))
    ncol, nrow = Base.size(grd)
    r = 1.0f0
    θ = LinRange(0.0f0, pi, nrow)
    φ = LinRange(0.0f0, 2f0*pi, ncol)
    x = [-(r + grd[i,j]) * cos(φ[i]) * sin(θ[j]) for i in 1:ncol, j in 1:nrow]
    y = [(r + grd[i,j]) * sin(φ[i]) * sin(θ[j]) for i in 1:ncol, j in 1:nrow]
    z = [(r + grd[i,j]) * cos(θ[j]) for i in 1:ncol, j in 1:nrow]
    return (; x,y,z)

end

function grid2surfpoints(::Val{:flat}, grd=fill(0.0, 180, 90))
    ncol, nrow = Base.size(grd)
    maxlength = norm([nrow, ncol])
    scaler = maxlength / 2π |> Float32


    x = [i - ncol/2f0 for i in 1:ncol, j in 1:nrow]
    y = [j - nrow/2f0 for i in 1:ncol, j in 1:nrow]
    z = [grd[i, j]*scaler for i in 1:ncol, j in 1:nrow]
    #print(maximum(z))
    return (; x, y, z)

end

function grid2surfpoints(::Val{:torus}, grd=fill(0.0, 180, 90); R=1f0, r=0.5f0, λ=1.) ## λ = 1.0 ==> Torus, λ =0.0 ==> flat surface, inbetween ==> inbetween ;-)
    ncol, nrow = Base.size(grd)
    θ = LinRange(0.0f0, 2f0*pi, nrow)
    φ = LinRange(0.0f0, 2f0*pi, ncol)

    x = [λ*(R + r*cos(θ[j])) * cos(φ[i]) + (1-λ) * φ[i] for i in 1:ncol, j in 1:nrow]
    y = [λ*(R + r*cos(θ[j])) * sin(φ[i]) + (1-λ) * θ[j] for i in 1:ncol, j in 1:nrow]
    z = [λ * r * sin(θ[j]) for i in 1:ncol, j in 1:nrow]

    return (; x, y, z)
end


function brick2Sphere(brick, layer; kwargs...)
    minmax_ = extrema(filter(isfinite, brick))
    matrix2Sphere(brick[:,:, layer]; minmax_=minmax_, kwargs...)
end



# Function to 
# Merge a merge a matrix representing a lat-lon map with values to be plotted with 
#   and elevation map (used to plot missing values)
#   and country borders 
function matrix2earth(br; typ=:flat, minmax_=nothing,
    pal=resample_cmap(ColorSchemes.speed, 100),
    smoothWin=Kernel.gaussian(2),
    #elev=nc.Dataset("D:/markusr/_dataMDI/Topo0.5deg.nc")["average.height.of.halfdegree.gridcell"][:] |> hrev,
    elev=dem05deg,
    value_height_max=0.2f0,
    elev_height_max=0.1f0, 
    overlay=countryRaster(;res=0.5f0), 
    kwargs...
)

    brSize=Base.size(br)
    #br = brick[:, :, layer]

    # Smooth Matrix using the mapwindow function with size smoothWin
    nok = .!isfinite.(br)
    #saveMean(x) = (xfilt = filter(isfinite, x); return isempty(xfilt) ? NaN : Statistics.mean(xfilt))
    #br = mapwindow(saveMean, br, smoothWin)  .|> Float32
    
    kern = smoothWin
    function gaussFilter(x)
        ok = findall(isfinite, x |> vec)
        return isempty(ok) ? NaN : Statistics.mean(x[ok], weights(kern)[ok])
    end
    br = mapwindow(gaussFilter, br, size(kern))  .|> Float32

    br[nok] .= NaN32
    # For better plottting missing values at a border to non-missing are set to minimum of non-missings
    filt(x) = any(isfinite.(x)) ? minimum(x) : NaN32
    brFilt = mapwindow(filt, br, (3, 3))
    br[nok] .= brFilt[nok]

    # Use of determine minmax of values
    minmax_Val = extrema(filter(isfinite, br))
    if isnothing(minmax_)
        minmax_ = minmax_Val
    end
    mini, maxi = minmax_

    # Resize elevation data and overlay to match size of value matrix
    if typ == :robin
        elev=proj(elev)
        overlay=nothing
    end
    elev = imresize(elev, brSize)

    # Define colors of value matrix. Missings set to Grey70 and where elev is missing to transparent (outside projection)
    cIdx(x) = @chain Int(1 + ((length(pal) - 1) * (x - mini)) ÷ (maxi - mini)) max(1) min(length(pal))
    valColors = map(x -> isfinite(x) ? pal[cIdx(x)] : RGBA(0.7, 0.7, 0.7, 1.0), br)
    valColors[.!isfinite.(elev)] .= RGBA(1,1,1,0)

    # Overlay is replacing colors with non-transparent black
    if !isnothing(overlay) 
        overlay = imresize(overlay, brSize, method=Lanczos4OpenCV())
        valColors[overlay .== 1] .= RGBA(0,0,0,1)
    end


    ## Calculate merged height of values and elevation matrix
    minmax_Elev = extrema(filter(isfinite, elev))
    #replace!(elev, NaN => 0f0)
    maxAbs = max(abs(mini), abs(maxi))
    br[nok] .= elev_height_max / maximum(abs.(minmax_Elev)) * elev[nok]
    br[.!nok] .= value_height_max / maxAbs * br[.!nok]
    #println(maximum(br[.!nok]))
    ## Add water (Oceans) where values are missing and evelation < 0
    iswater = nok .& (elev .< 0)
    #iswater = mapwindow(any, iswater, [5,5])
    waterColor = map(iswater) do x
        return ifelse(x, (:lightblue, 0.8),(:transparent))
    end
    eltype(br) != Float32 &&  @warn("not Float32 at matrix2earth")

    return (; heights = br, shades = valColors, iswater, colorRange=minmax_, pal=pal )
    # brp = grid2surfPoints(:sphere, br)
    # water = grid2surfPoints(:sphere, ifelse.(iswater, 0.0, 0.0))
    # return (; surfcoords=brp, valueColors=valColors, water=water, waterColor=waterColor, colorRange=minmax_, pal=pal, br)

end

function matrix2sphere(br, typ=:sphere; kwargs...)
    earth = matrix2earth(br; typ=typ, kwargs...)
    if typ == :robin typ=:flat end
    waterColor = map(earth.iswater) do x
        return ifelse(x, (:lightblue, 0.8),(:transparent))
    end
    return (; surfcoords=grid2surfpoints(Val(typ), earth.heights), 
            water = grid2surfpoints(Val(typ), ifelse.(earth.iswater, 0.0, 0.0)),
            valueColors=earth.shades, 
            waterColor = waterColor, 
            colorRange=earth.colorRange, pal=earth.pal
            )
end

function brick2Sphere_withEta(brick, layer,  η, typ=:sphere; kwargs...)
    nr, nc, nlayers = Base.size(brick)
    matr = @. (1 - η) * brick[:, :, layer] + η * brick[:,:, layer % nlayers + 1]
    if typ == :robin
        matr=proj(matr)
    end
    return matrix2sphere(matr, typ;  kwargs...)
end

# function brick2movie(brick; nsubSteps=1, nRounds=1, kwargs...)
#     nr, nc, nlayers = Base.size(brick)
#     minmax_ = extrema(filter(isfinite, brick))
  
function brick2movie(brick, typ::Symbol=:sphere; nsubsteps=4, nRounds=10,
    theta_traj=typ == :sphere ? [0, 360] : [0,0], phi_traj=[45, 45],
    layerTitle=i -> monthname(i), title="GPP mean seasonal cycle",
    outfile="video.mp4", resolution=(1920, 1080),
    legLabel=L"gC\,\, m^{-2} day^{-1}", kwargs...)

    nlayers = Base.size(brick)[3]
    nFrames = nlayers * nsubsteps * nRounds
    #θ_all = range(theta_traj..., nFrames+1)
    itp = Interpolations.interpolate(theta_traj, BSpline(Linear()))
    θ_all = itp(range(1, length(theta_traj), nFrames + 1))
    itp = Interpolations.interpolate(phi_traj, BSpline(Linear()))
    ϕ_all = itp(range(1, length(phi_traj), nFrames + 1))
    #ϕ_all = range(phi_traj..., nFrames+1)
    idx = Observable(1) # the index of the layer
    relSubStep = Observable(0.0f0)

    res = @lift(brick2Sphere_withEta(brick, $idx, $relSubStep, typ;
        minmax_=extrema(filter(isfinite, brick)), kwargs...))
    println(typeof(res[].surfcoords.z))
    
    GLMakie.activate!()

    #fig = Scene(resolution=1.5 .* (1920,1080))
    #campixel!(fig)
    fig = Figure(; resolution)
    axis = LScene(fig[1, 1], show_axis=false)
    cam3d!(axis.scene)
    #Box(fig[1,1])

    s1 = surface!(axis, @lift($res[:surfcoords][1]),
        @lift($res[:surfcoords][2]),
        @lift($res[:surfcoords][3]),
        color=@lift($res[:valueColors]), invert_normals=false)
    s2 = surface!(axis, @lift($res[:water][1]),
        @lift($res[:water][2]),
        @lift($res[:water][3]), color=res[].waterColor, invert_normals=false) #=fill((:lightblue, 0.8), Base.size(brick[:,:,idx[]])),=#


    Label(fig[1, 1, Top()], @lift(layerTitle($idx)), padding=(0, 0, 5, 0), fontsize=50.0f0)
    Label(fig[1, 1:2, Bottom()], title, padding=(0, 0, 5, 0), fontsize=40.0f0, halign=:right)
    Colorbar(fig[1, 2], colorrange=res[].colorRange, colormap=res[].pal, label=legLabel,
        height=Relative(0.8), width=30, labelsize=32, ticklabelsize=32)
    #rot = Observable(π)
    if typ == :sphere
        #GLMakie.rotate!(axis.scene, Vec3f(0, 0, 1), π)
        GLMakie.zoom!(axis.scene, 0.7)
        rotate_cam!(axis.scene, (1.1π, 0.5π, 0.0))

    else
        #GLMakie.rotate!(axis.scene, Vec3f(0, 0, 1), 0)
        GLMakie.zoom!(axis.scene, 0.7)
        rotate_cam!(axis.scene, (-360/16 / 360 * 2π, 225 / 360 * 2π,- 0.0))
    end

    #rotate_cam!(axis.scene, (0.5,0,0))
    #GLMakie.rotate!(Accum, axis.scene, Vec3f(0, 1, 0), π/4 )

    display(fig)
    sleep(1.0)
    list = [(; r, layer, sub) for sub in 0:nsubsteps-1, layer in 1:nlayers, r in 0:nRounds-1]
    #println(list)
    list = ProgressBar(list)
    i = 1
    #anim = @time GLMakie.record(fig, outfile, list, framerate=16) do e
    anim = @time GLMakie.Record(fig, list, framerate=16) do e
        idx[] = e.layer
        relSubStep[] = e.sub / nsubsteps
        # GLMakie.rotate!( axis.scene, Vec3f(0, 0, 1), π + 2π*(e.r*nlayers*nSubsteps + (e.layer-1) * nSubsteps + e.sub )/(nlayers*nSubsteps*nRounds) )
        #if typ == :sphere
            rotate_cam!(axis.scene, (0.0, -(θ_all[i] - θ_all[max(i-1,1)]) / 360 * 2π, 0.0))
            rotate_cam!(axis.scene, ((ϕ_all[i] - ϕ_all[max(i-1,1)]) / 360 * 2π, 0, 0))
        #end

        #GLMakie.rotate!(Accum,  axis.scene, Vec3f(0, 0, 1), (θ_all[i+1] - θ_all[i])/360*2π )
        #GLMakie.rotate!(Accum, axis.scene, Vec3f(0, 1, 0), ϕ_all[i]/360*2π )
        #println(θ_all[i], " ", e)
        i += 1
        #GLMakie.rotate!(fig.content[1].scene, Vec3f(0, 0, 1), i*π/48)
    end
    display(fig)
    save(outfile, anim)
    anim
end

##
function exampleDiurnal()
    ds = open_dataset(zopen("C:/Users/mreichstein/Desktop/_FLUXCOM/COCO2_NEE/NEE_quarter/", fill_as_missing=true))
    dsSub = ds[Time=DateTime(2019,6,1,12,0,0)]

    NEE = @chain begin
        dsSub["NEE_quarter"][:,:,:]
        coalesce.(NaN32)
        imresize(ratio=(0.5, 0.5, 1))
        reverse(dims=2)
    end

    #NEE = NEE[:,:, 1:3]
    title="Net CO_2 flux diurnal cycle June 2019 (FLUXCOM @ MPI-BGC)"
    legLabel = L"[µmol\,\, m^{-2} s^{-1}]"
    layerTitle(i) = string("Time: ", Time(i-1, 0,0) + Hour(4))
    pal=create_cmap(sharpen_colorscheme(ColorSchemes.BrBg) |> ColorScheme, -20, 10)
    plot(s::Symbol) = brick2movie(circshift(NEE, (0,0,-4)), s; nRounds=1, nsubsteps=20,  elev=dem05,  #overlay=countryRaster(;res=0.25f0), smoothWin=Kernel.gaussian(3),
        pal, title, legLabel, minmax_=[-20,10], layerTitle, outfile="NEE_video_$s.mp4")

    plot(:sphere)
    plot(:flat)
    plot(:robin)

    
end

function exampleSeasonal()
    ds = nc.Dataset("C:/Users\\mreichstein\\Desktop\\_FLUXCOM\\0d50_monthly\\GPP_MR.msc.nc")
    gpp=@chain begin
        ds[:variable][:]
        coalesce.(NaN32)
        reverse(dims=2)
    end 

    gpp = imresize(gpp,ratio=(2, 2, 1))


    title="Gross primary production mean season cycle (FLUXCOM @ MPI-BGC)"
    legLabel = L"[gC\,\, m^{-2} day^{-1}]"
    layerTitle(i) = monthname(i) # which is the default
    pal=resample_cmap(ColorSchemes.speed, 100) # which is default
    plot(s::Symbol) = brick2movie(gpp, s; nRounds=10, nsubsteps=4,  elev=dem05deg, overlay=countryRaster(;res=0.25f0), smoothWin=Kernel.gaussian(3),
        pal, title, legLabel, layerTitle, outfile="GPP025_seas_video_$s.mp4")

    plot(:sphere)
   # plot(:flat)
    #plot(:robin)
end

function quickRendercheck(file)
    ds = nc.Dataset(file)
    var = findfirst(x->ndims(x.second)>1, ds)
    dat=@chain ds[var][:] coalesce.(NaN32) Float32.()  reverse(dims=2) imresize(ratio=(2, 2, 1))
    outfile = basename(file) |> splitext |> first
    title=outfile
    legLabel = ""
    layerTitle(i) = "Layer $i" #
    pltype=:robin
    pal=resample_cmap(ColorSchemes.speed, 100) # which is default

    #brick2movie(dat[:,:,1:2], pltype; nRounds=1, nsubsteps=4,  elev=dem05deg, overlay=countryRaster(;res=0.25f0), smoothWin=Kernel.gaussian(3),
    #    pal, title, legLabel, layerTitle, outfile="$(outfile)_test.mp4")
    dat

end





