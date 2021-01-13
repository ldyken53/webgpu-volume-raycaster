(async () => {
    function makeBufferRequest(method, url) {
        return new Promise(function (resolve, reject) {
            let xhr = new XMLHttpRequest();
            xhr.open(method, url);
            xhr.responseType = "arraybuffer";
            xhr.onload = function () {
                if (this.status >= 200 && this.status < 300) {
                    dataBuffer = xhr.response;
                    if (dataBuffer) {
                        dataBuffer = new Uint8Array(dataBuffer);
                    } else {
                        alert("Unable to load buffer properly from volume?");
                        console.log("no buffer?");
                    }
                    resolve(xhr.response);
                } else {
                    reject({
                        status: this.status,
                        statusText: xhr.statusText
                    });
                }
            };
            xhr.onerror = function () {
                reject({
                    status: this.status,
                    statusText: xhr.statusText
                });
            };
            xhr.send();
        });
    }

    var dataBuffer = null;
    var adapter = await navigator.gpu.requestAdapter();

    var device = await adapter.requestDevice();

    var canvas = document.getElementById("webgpu-canvas");
    var context = canvas.getContext("gpupresent");
    var original = [];

    var fileRegex = /(\w+)_(\d+)x(\d+)x(\d+)_(\w+)\.*/;

    var getVolumeDimensions = function (name) {
        var m = name.match(fileRegex);
        return [parseInt(m[2]), parseInt(m[3]), parseInt(m[4])];
    }

    var datasets = {
        fuel: {
            name: "7d87jcsh0qodk78/fuel_64x64x64_uint8.raw",
            range: [0, 255],
            scale: [1, 1, 1]
        }
    }

    var dataset = datasets.fuel;
    if (window.location.hash) {
        var name = decodeURI(window.location.hash.substr(1));
        console.log(`Linked to data set ${name}`);
        dataset = datasets[name];
    }

    var url = "https://www.dl.dropboxusercontent.com/s/" + dataset.name + "?dl=1";
    await makeBufferRequest("GET", url);

    var volumeDims = getVolumeDimensions(dataset.name);

    // Setup shader modules
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
    var dataBuf = device.createBuffer({
        size: 12 * 3 * 3 * 4,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    new Float32Array(dataBuf.getMappedRange()).set([
        1, 0, 0,
        0, 0, 0,
        1, 1, 0,

        0, 1, 0,
        1, 1, 0,
        0, 0, 0,

        1, 0, 1,
        1, 0, 0,
        1, 1, 1,

        1, 1, 0,
        1, 1, 1,
        1, 0, 0,

        0, 0, 1,
        1, 0, 1,
        0, 1, 1,

        1, 1, 1,
        0, 1, 1,
        1, 0, 1,

        0, 0, 0,
        0, 0, 1,
        0, 1, 0,

        0, 1, 1,
        0, 1, 0,
        0, 0, 1,

        1, 1, 0,
        0, 1, 0,
        1, 1, 1,

        0, 1, 1,
        1, 1, 1,
        0, 1, 0,

        0, 0, 1,
        0, 0, 0,
        1, 0, 1,

        1, 0, 0,
        1, 0, 1,
        0, 0, 0
    ]);
    dataBuf.unmap();

    var colorBuf = device.createBuffer({
        size: 12 * 3 * 3 * 4,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    new Float32Array(colorBuf.getMappedRange()).set([
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,

        1, 0, 0,
        1, 0, 0,
        1, 0, 0,
        1, 0, 0,
        1, 0, 0,
        1, 0, 0,

        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,

        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,

        1, 1, 0,
        1, 1, 0,
        1, 1, 0,
        1, 1, 0,
        1, 1, 0,
        1, 1, 0,

        0, 1, 1,
        0, 1, 1,
        0, 1, 1,
        0, 1, 1,
        0, 1, 1,
        0, 1, 1
    ]);
    colorBuf.unmap();

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
            },
            {
                arrayStride: 3 * 4,
                attributes: [
                    {
                        format: "float3",
                        offset: 0,
                        shaderLocation: 1
                    }
                ]
            }
        ]
    };

    // Setup render outputs
    var swapChainFormat = "bgra8unorm";
    var swapChain = context.configureSwapChain({
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

    // Create the bind group layout
    var bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                type: "uniform-buffer"
            },
            {
                binding: 1,
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
                type: "sampler"
            },
        ]
    });

    // Create render pipeline
    var layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    var renderPipeline = device.createRenderPipeline({
        layout: layout,
        vertexStage: vertexStage,
        fragmentStage: fragmentStage,
        primitiveTopology: "triangle-list",
        rasterizationState: {
            cullMode: "front",
        },
        vertexState: vertexState,
        colorStates: [{
            format: swapChainFormat,
            alphaBlend: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
            },
        }],
        depthStencilState: {
            format: depthFormat,
            depthWriteEnabled: true,
            depthCompare: "less"
        }
    });

    var renderPassDesc = {
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

    // Load the default colormap and upload it
    var colormapImage = new Image();
    colormapImage.src = "colormaps/cool-warm-paraview.png";
    await colormapImage.decode();
    const imageBitmap = await createImageBitmap(colormapImage);
    var colorTexture = device.createTexture({
        size: [imageBitmap.width, imageBitmap.height, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.SAMPLED | GPUTextureUsage.COPY_DST,
    });
    device.defaultQueue.copyImageBitmapToTexture(
        { imageBitmap }, { texture: colorTexture },
        [imageBitmap.width, imageBitmap.height, 1]
    );

    // Create our sampler
    const sampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear"
    });

    // Create a buffer to store the view parameters
    var viewParamsBuffer = device.createBuffer({
        size: 20 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    var volumeDataBuffer = device.createBuffer({
        size: 64 * 64 * 64 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    // Create a bind group which places our view params buffer at binding 0
    var bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: viewParamsBuffer,
                    size: 20 * 4,
                    offset: 0
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: volumeDataBuffer,
                    size: 64 * 64 * 64 * 4,
                    offset: 0
                }
            },
            {
                binding: 2,
                resource: colorTexture.createView(),
            },
            {
                binding: 3,
                resource: sampler,
            },
        ]
    });

    // Debug function
    function pos(arr) {
        return arr.reduce((ret_arr, number, index) => {
            if (number > 0) ret_arr.push(index)
            return ret_arr
        }, [])

    }

    // Buffer the volume data
    var upload = device.createBuffer({
        size: 64 * 64 * 64 * 4,
        usage: GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true
    });
    {
        var map = new Uint32Array(upload.getMappedRange());
        map.set(dataBuffer);
    }
    upload.unmap();

    var commandEncoder = device.createCommandEncoder();

    // Copy the upload buffer to our storage buffer
    commandEncoder.copyBufferToBuffer(upload, 0, volumeDataBuffer, 0, 64 * 64 * 64 * 4);
    device.defaultQueue.submit([commandEncoder.finish()]);

    // Create an arcball camera and view projection matrix
    var camera = new ArcballCamera([0, 0, 3], [0, 0, 0], [0, 1, 0],
        0.5, [canvas.width, canvas.height]);
    var projection = mat4.perspective(mat4.create(), 50 * Math.PI / 180.0,
        canvas.width / canvas.height, 0.1, 100);
    // Matrix which will store the computed projection * view matrix
    var projView = mat4.create();

    // Controller utility for interacting with the canvas and driving
    // the arcball camera
    var controller = new Controller();
    controller.mousemove = function (prev, cur, evt) {
        if (evt.buttons == 1) {
            camera.rotate(prev, cur);

        } else if (evt.buttons == 2) {
            camera.pan([cur[0] - prev[0], prev[1] - cur[1]]);
        }
    };
    controller.wheel = function (amt) { camera.zoom(amt * 0.5); };
    controller.registerForCanvas(canvas);

    // Not covered in the tutorial: track when the canvas is visible
    // on screen, and only render when it is visible.
    var canvasVisible = false;
    var observer = new IntersectionObserver(function (e) {
        if (e[0].isIntersecting) {
            canvasVisible = true;
        } else {
            canvasVisible = false;
        }
    }, { threshold: [0] });
    observer.observe(canvas);

    var animationFrame = function () {
        var resolve = null;
        var promise = new Promise(r => resolve = r);
        window.requestAnimationFrame(resolve);
        return promise
    };

    requestAnimationFrame(animationFrame);

    while (true) {
        await animationFrame();

        if (canvasVisible) {
            renderPassDesc.colorAttachments[0].attachment =
                swapChain.getCurrentTexture().createView();

            // Upload the combined projection and view matrix
            projView = mat4.mul(projView, projection, camera.camera);
            var upload = device.createBuffer({
                size: 20 * 4,
                usage: GPUBufferUsage.COPY_SRC,
                mappedAtCreation: true
            });
            {
                var map = new Float32Array(upload.getMappedRange());
                map.set(projView);
                map.set(camera.eyePos(), 16)
            }
            upload.unmap();

            var commandEncoder = device.createCommandEncoder();

            // Copy the upload buffer to our uniform buffer
            commandEncoder.copyBufferToBuffer(upload, 0, viewParamsBuffer, 0, 20 * 4);

            var renderPass = commandEncoder.beginRenderPass(renderPassDesc);

            renderPass.setPipeline(renderPipeline);
            renderPass.setVertexBuffer(0, dataBuf);
            renderPass.setVertexBuffer(1, colorBuf);
            renderPass.setBindGroup(0, bindGroup);
            renderPass.draw(12 * 3, 1, 0, 0);

            renderPass.endPass();
            device.defaultQueue.submit([commandEncoder.finish()]);
        }
    }
})();


