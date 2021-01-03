var cubeStrip = [
    1, 1, 0,
    0, 1, 0,
    1, 1, 1,
    0, 1, 1,
    0, 0, 1,
    0, 1, 0,
    0, 0, 0,
    1, 1, 0,
    1, 0, 0,
    1, 1, 1,
    1, 0, 1,
    0, 0, 1,
    1, 0, 0,
    0, 0, 0
];

var takeScreenShot = false;
var canvas = null;

var device = null;
var adapter = null;
var dataBuf = null;
var volumeDataBuffer = null;
var volumeParamsBuffer = null;
var viewParamsBuffer = null;
var colorTexture = null;
var renderPipeline = null;
var renderPassDesc = null;
var swapChain = null;
var fileRegex = /.*\/(\w+)_(\d+)x(\d+)x(\d+)_(\w+)\.*/;
var proj = null;
var camera = null;
var projView = null;
var tabFocused = true;
var newVolumeUpload = true;
var targetFrameTime = 32;
var samplingRate = 1.0;
var WIDTH = 640;
var HEIGHT = 480;

const defaultEye = vec3.set(vec3.create(), 0.5, 0.5, 1.5);
const center = vec3.set(vec3.create(), 0.5, 0.5, 0.5);
const up = vec3.set(vec3.create(), 0.0, 1.0, 0.0);

var volumes = {
    "Fuel": "7d87jcsh0qodk78/fuel_64x64x64_uint8.raw",
    "Neghip": "zgocya7h33nltu9/neghip_64x64x64_uint8.raw",
    "Hydrogen Atom": "jwbav8s3wmmxd5x/hydrogen_atom_128x128x128_uint8.raw",
    "Boston Teapot": "w4y88hlf2nbduiv/boston_teapot_256x256x178_uint8.raw",
    "Engine": "ld2sqwwd3vaq4zf/engine_256x256x128_uint8.raw",
    "Bonsai": "rdnhdxmxtfxe0sa/bonsai_256x256x256_uint8.raw",
    "Foot": "ic0mik3qv4vqacm/foot_256x256x256_uint8.raw",
    "Skull": "5rfjobn0lvb7tmo/skull_256x256x256_uint8.raw",
    "Aneurysm": "3ykigaiym8uiwbp/aneurism_256x256x256_uint8.raw",
};

var colormaps = {
    "Cool Warm": "colormaps/cool-warm-paraview.png",
    "Matplotlib Plasma": "colormaps/matplotlib-plasma.png",
    "Matplotlib Virdis": "colormaps/matplotlib-virdis.png",
    "Rainbow": "colormaps/rainbow.png",
    "Samsel Linear Green": "colormaps/samsel-linear-green.png",
    "Samsel Linear YGB 1211G": "colormaps/samsel-linear-ygb-1211g.png",
};

var loadVolume = function (file, onload) {
    var m = file.match(fileRegex);
    var volDims = [parseInt(m[2]), parseInt(m[3]), parseInt(m[4])];

    var url = "https://www.dl.dropboxusercontent.com/s/" + file + "?dl=1";
    var req = new XMLHttpRequest();
    var loadingProgressText = document.getElementById("loadingText");
    var loadingProgressBar = document.getElementById("loadingProgressBar");

    loadingProgressText.innerHTML = "Loading Volume";
    loadingProgressBar.setAttribute("style", "width: 0%");

    req.open("GET", url, true);
    req.responseType = "arraybuffer";
    req.onprogress = function (evt) {
        var vol_size = volDims[0] * volDims[1] * volDims[2];
        var percent = evt.loaded / vol_size * 100;
        loadingProgressBar.setAttribute("style", "width: " + percent.toFixed(2) + "%");
    };
    req.onerror = function (evt) {
        loadingProgressText.innerHTML = "Error Loading Volume";
        loadingProgressBar.setAttribute("style", "width: 0%");
    };
    req.onload = function (evt) {
        loadingProgressText.innerHTML = "Loaded Volume";
        loadingProgressBar.setAttribute("style", "width: 100%");
        var dataBuffer = req.response;
        if (dataBuffer) {
            dataBuffer = new Uint8Array(dataBuffer);
            onload(file, dataBuffer);
        } else {
            alert("Unable to load buffer properly from volume?");
            console.log("no buffer?");
        }
    };
    req.send();
}

var selectVolume = function () {
    var selection = document.getElementById("volumeList").value;
    history.replaceState(history.state, "#" + selection, "#" + selection);

    loadVolume(volumes[selection], function (file, dataBuffer) {
        var m = file.match(fileRegex);
        var volDims = [parseFloat(m[2]), parseFloat(m[3]), parseFloat(m[4])];

        var longestAxis = Math.max(volDims[0], Math.max(volDims[1], volDims[2]));
        var volScale = [volDims[0] / longestAxis, volDims[1] / longestAxis,
        volDims[2] / longestAxis];
        // Upload the volume data
        upload = device.createBuffer({
            size: 64 * 64 * 64 * 1,
            usage: GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true
        })
        new Uint8Array(upload.getMappedRange()).set(dataBuffer);
        upload.unmap();

        var commandEncoder = device.createCommandEncoder();

        commandEncoder.copyBufferToBuffer(upload, 0, volumeDataBuffer, 0, 64 * 64 * 64 * 1);

        newVolumeUpload = true;
        setInterval(function () {
            // Save them some battery if they're not viewing the tab
            if (document.hidden) {
                return;
            }

            // Reset the sampling rate and camera for new volumes
            if (newVolumeUpload) {
                camera = new ArcballCamera(defaultEye, center, up, 2, [WIDTH, HEIGHT]);
            }
            renderPassDesc.colorAttachments[0].attachment = swapChain.getCurrentTexture().createView();

            // Compute and upload the combined projection and view matrix
            projView = mat4.mul(projView, proj, camera.camera);
            var eye = [camera.invCamera[12], camera.invCamera[13], camera.invCamera[14]];
            var upload = device.createBuffer({
                size: 16 * 4 + 2 * 3 * 4,
                usage: GPUBufferUsage.COPY_SRC,
                mappedAtCreation: true
            });
            var map = new Float32Array(upload.getMappedRange());
            map.set(projView);
            map.set(eye, 16);
            map.set(volScale, 19);
            upload.unmap();

            // Compute and upload the volume params
            var test = device.createBuffer({
                size: 3 * 4 + 4,
                usage: GPUBufferUsage.COPY_SRC,
                mappedAtCreation: true
            })
            // TODO have to separate volume dims because float vs int
            var map = new Float32Array(test.getMappedRange());
            map.set(volDims);
            map.set([samplingRate], 3);
            upload.unmap();

            var commandEncoder = device.createCommandEncoder();

            commandEncoder.copyBufferToBuffer(upload, 0, viewParamsBuffer, 0, 16 * 4 + 2 * 3 * 4);
            commandEncoder.copyBufferToBuffer(upload, 0, volumeParamsBuffer, 0, 3 * 4 + 4);

            var renderPass = commandEncoder.beginRenderPass(renderPassDesc);

            renderPass.setPipeline(renderPipeline);
            renderPass.setVertexBuffer(0, dataBuf);
            renderPass.draw(cubeStrip.length / 3, 1, 0, 0);

            renderPass.endPass();
            device.defaultQueue.submit([commandEncoder.finish()]);

            newVolumeUpload = false;
        }, targetFrameTime);
    });
}

var selectColormap = function () {
    var selection = document.getElementById("colormapList").value;
    var colormapImage = new Image();
    colormapImage.onload = function () {
        gl.activeTexture(gl.TEXTURE1);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 180, 1,
            gl.RGBA, gl.UNSIGNED_BYTE, colormapImage);
    };
    colormapImage.src = colormaps[selection];
}

var fillVolumeSelector = function () {
    var selector = document.getElementById("volumeList");
    for (v in volumes) {
        var opt = document.createElement("option");
        opt.value = v;
        opt.innerHTML = v;
        selector.appendChild(opt);
    }
}

var fillcolormapSelector = function () {
    var selector = document.getElementById("colormapList");
    for (p in colormaps) {
        var opt = document.createElement("option");
        opt.value = p;
        opt.innerHTML = p;
        selector.appendChild(opt);
    }
}

window.onload = async function () {
    fillVolumeSelector();
    fillcolormapSelector();

    if (!navigator.gpu) {
        alert("WebGPU is not supported/enabled in your browser");
        return;
    }

    // Get a GPU device to render with
    adapter = await navigator.gpu.requestAdapter();
    device = await adapter.requestDevice();

    // Get a context to display our rendered image on the canvas
    var canvas = document.getElementById("webgpu-canvas");
    var context = canvas.getContext("gpupresent");

    // Setup camera controls
    proj = mat4.perspective(mat4.create(), 60 * Math.PI / 180.0,
        canvas.width / canvas.height, 0.1, 100);

    camera = new ArcballCamera(defaultEye, center, up, 2, [WIDTH, HEIGHT]);
    projView = mat4.create();

    // Register mouse and touch listeners
    var controller = new Controller();
    controller.mousemove = function (prev, cur, evt) {
        if (evt.buttons == 1) {
            camera.rotate(prev, cur);

        } else if (evt.buttons == 2) {
            camera.pan([cur[0] - prev[0], prev[1] - cur[1]]);
        }
    };
    controller.wheel = function (amt) { camera.zoom(amt); };
    controller.pinch = controller.wheel;
    controller.twoFingerDrag = function (drag) { camera.pan(drag); };

    document.addEventListener("keydown", function (evt) {
        if (evt.key == "p") {
            takeScreenShot = true;
        }
    });

    controller.registerForCanvas(canvas);

    var vertModule = device.createShaderModule({ code: simple_vert_spv });
    var vertexStage = {
        module: vertModule,
        entryPoint: "main"
    };

    var fragModule = device.createShaderModule({ code: simple_frag_spv });
    var fragmentStage = {
        module: fragModule,
        entryPoint: "main"
    };

    // Specify vertex data
    dataBuf = device.createBuffer({
        size: 14 * 3 * 4,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    // Vertex positions
    new Float32Array(dataBuf.getMappedRange()).set(cubeStrip);
    dataBuf.unmap();

    var vertexState = {
        vertexBuffers: [
            {
                arrayStride: 3 * 4,
                attributes: [
                    {
                        format: "float3",
                        offset: 0,
                        shaderLocation: 0
                    }
                ]
            }
        ]
    };

    // Setup render outputs
    var swapChainFormat = "bgra8unorm";
    swapChain = context.configureSwapChain({
        device: device,
        format: swapChainFormat,
        usage: GPUTextureUsage.OUTPUT_ATTACHMENT
    });

    var depthFormat = "depth24plus-stencil8";
    var depthTexture = device.createTexture({
        size: {
            width: canvas.width,
            height: canvas.height,
            depth: 1
        },
        format: depthFormat,
        usage: GPUTextureUsage.OUTPUT_ATTACHMENT
    });

    // Create bind group layout
    var bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                // One or more stage flags, or'd together
                visibility: GPUShaderStage.VERTEX,
                type: "uniform-buffer"
            },
            {
                binding: 1,
                // One or more stage flags, or'd together
                visibility: GPUShaderStage.FRAGMENT,
                type: "storage-buffer"
            },
            {
                binding: 2,
                // One or more stage flags, or'd together
                visibility: GPUShaderStage.FRAGMENT,
                type: "sampled-texture"
            },
            {
                binding: 3,
                // One or more stage flags, or'd together
                visibility: GPUShaderStage.FRAGMENT,
                type: "uniform-buffer"
            },
            {
                binding: 4,
                // One or more stage flags, or'd together
                visibility: GPUShaderStage.FRAGMENT,
                type: "sampler"
            },
        ]
    });

    // Create the pipeline layout, specifying which bind group layouts will be used
    var layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    // Create a buffer to store the volume data
    volumeDataBuffer = device.createBuffer({
        size: 64 * 64 * 64 * 1,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    // Create a buffer to store the volume parameters
    volumeParamsBuffer = device.createBuffer({
        size: 3 * 4 + 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Create a buffer to store the view parameters
    viewParamsBuffer = device.createBuffer({
        size: 16 * 4 + 2 * 3 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Load the default colormap and upload it
    var colormapImage = new Image();
    colormapImage.src = "colormaps/cool-warm-paraview.png";
    await colormapImage.decode();
    const imageBitmap = await createImageBitmap(colormapImage);
    colorTexture = device.createTexture({
        size: [imageBitmap.width, imageBitmap.height, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.SAMPLED | GPUTextureUsage.COPY_DST,
    });
    device.defaultQueue.copyImageBitmapToTexture(
        { imageBitmap }, { texture: colorTexture },
        [imageBitmap.width, imageBitmap.height, 1]
    );

    const sampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear"
    });

    // Create a bind group which places our view params buffer at binding 0
    bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: viewParamsBuffer
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: volumeDataBuffer
                },
            },
            {
                binding: 2,
                resource: colorTexture.createView(),
            },
            {
                binding: 3,
                resource: {
                    buffer: volumeParamsBuffer
                },
            },
            {
                binding: 4,
                resource: sampler,
            },
        ]
    });

    // Create render pipeline
    renderPipeline = device.createRenderPipeline({
        layout: layout,
        vertexStage: vertexStage,
        fragmentStage: fragmentStage,
        primitiveTopology: "triangle-list",
        vertexState: vertexState,
        colorStates: [{
            format: swapChainFormat
        }],
        depthStencilState: {
            format: depthFormat,
            depthWriteEnabled: true,
            depthCompare: "less"
        }
    });

    renderPassDesc = {
        colorAttachments: [{
            attachment: undefined,
            loadValue: [0.3, 0.3, 0.3, 1]
        }],
        depthStencilAttachment: {
            attachment: depthTexture.createView(),
            depthLoadValue: 1.0,
            depthStoreOp: "store",
            stencilLoadValue: 0,
            stencilStoreOp: "store"
        }
    };

    selectVolume();
    //TODO add the functionality to not render when canvas is off screen
}