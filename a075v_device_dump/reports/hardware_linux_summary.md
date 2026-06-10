# Sipeed A075V Hardware/Linux Findings

Raw reports:
- `a075v_device_dump/reports/hardware_linux_kernel_report.txt`
- `a075v_device_dump/reports/hardware_interfaces_followup.txt`

## Platform

- SoC family: Allwinner `sun8iw19p1`.
- Tina/OpenWrt target: `v833-perf1/generic v2.1`.
- CPU: single ARMv7 Cortex-A7 class core, `/proc/cpuinfo` reports `ARMv7 Processor rev 5`, hardware `sun8iw19`.
- RAM: about `124540 kB`; CMA reserved: `16M`.
- Device tree model: `sun8iw19`; compatible: `allwinner,sun8iw19p1`.
- Bootloader reported in kernel cmdline: U-Boot `2018.05(05/31/2021-20:25:24)`.

## Linux Image

- Distro: Tina Linux / OpenWrt-derived.
- `/etc/openwrt_release`:
  - `DISTRIB_ID='tina.work.20230109.013343'`
  - `DISTRIB_RELEASE='Neptune'`
  - `DISTRIB_REVISION='5C1C9C53'`
  - `DISTRIB_TARGET='v833-perf1/generic v2.1'`
  - `DISTRIB_DESCRIPTION='tina.work.20230109.013343 3.5.1'`
- Kernel: `Linux 4.9.118 #3410 PREEMPT Mon Jan 9 01:34:54 UTC 2023 armv7l`.
- Kernel toolchain: `OpenWrt/Linaro GCC 6.4-2017.11`.
- Root filesystem: ext4 on `/dev/mmcblk0p5`.
- Application/user partition: ext4 `/dev/mmcblk0p6`, mounted at both `/mnt/UDISK` and `/root`.
- Storage detected by kernel: `mmcblk0: mmc0:21b3 EASTC 121 MiB`.

## Partition Layout

From kernel cmdline and `/dev/by-name`:

- `bootlogo` -> `/dev/mmcblk0p1`, about `3048 KiB`.
- `cfg` -> `/dev/mmcblk0p2`, about `5120 KiB`, mounted at `/mnt/cfg`.
- `env` -> `/dev/mmcblk0p3`, about `256 KiB`.
- `boot` -> `/dev/mmcblk0p4`, about `6144 KiB`.
- `rootfs` -> `/dev/mmcblk0p5`, about `65536 KiB`, mounted as `/`.
- `UDISK` -> `/dev/mmcblk0p6`, about `19207 KiB`, mounted as `/mnt/UDISK` and `/root`.

## Loaded Relevant Kernel Modules

- `vin_v4l2`: Allwinner video input V4L2 driver; depends on `vin_io` and `videobuf2-dma-contig`.
- `vin_io`: Allwinner sunxi video front-end, CSI/CCI camera interface support.
- `videobuf2_dma_contig`: contiguous DMA buffers for V4L2 capture.
- `sensor_power`: sensor power control.
- `opn87xx_mipi`: low-level I2C/MIPI ToF sensor driver, alias `i2c:opn87xx_mipi`.
- `gc2145`: low-level GalaxyCore RGB sensor driver, alias `i2c:gc2145`.
- `8723ds` and `8189fs`: Realtek SDIO Wi-Fi drivers; loaded even though the product is currently using USB Ethernet.

## Available Sensor/Hardware Modules

Additional modules present under `/lib/modules/4.9.118` include:

- ToF/depth candidates: `opn87xx_mipi.ko`, `irs2381c_mipi.ko`.
- RGB/image sensor candidates: `gc2145.ko`, `gc2093_mipi.ko`, `sp2305_mipi.ko`, `ov7251_mipi.ko`, `ov9732_mipi.ko`.
- Touch/UI: `goodix.ko`.
- USB/network: `rndis_host.ko`, `xhci-hcd.ko`, `xhci-plat-hcd.ko`, PPP/tunnel modules.
- Wi-Fi: `8189fs.ko`, `8723ds.ko`, `bl_fdrv.ko`.

The boot script `/etc/init.d/S00mpp` loads the camera stack in this order:

```sh
videobuf2-core.ko
videobuf2-memops.ko
videobuf2-dma-contig.ko
videobuf2-v4l2.ko
vin_io.ko
sensor_power.ko
opn87xx_mipi.ko
gc2145.ko
vin_v4l2.ko
```

Only some of these are standalone `.ko` files on the live filesystem; the missing `videobuf2-*` pieces are likely built into the kernel or stored elsewhere in the vendor image.

## Camera/Media Interfaces

- Main media device: `/dev/media0`.
- V4L capture nodes: `/dev/video0`, `/dev/video1`, `/dev/video2`, `/dev/video3`.
- `testisp` strings explicitly reference `/dev/video0` and `/dev/video2`.
- V4L subdevices include:
  - `v4l-subdev0`: `opn87xx_mipi`.
  - `v4l-subdev1`: `gc2145`.
  - `v4l-subdev2..5`: `vin_cap.0..3`.
  - `v4l-subdev6..7`: `sunxi_csi.0..1`.
  - `v4l-subdev8`: `sunxi_mipi.0`.
  - `v4l-subdev9` and `v4l-subdev11`: `sunxi_isp.0..1`.
  - `v4l-subdev10` and `v4l-subdev12`: `sunxi_h3a.0..1`.
  - `v4l-subdev13..16`: `sunxi_scaler.0..3`.
- Platform devices include `6600800.vind`, `660c000.mipi`, `csi0`, `csi1`, `vinc0..3`, `2100000.isp`, `2104000..2104c00.scaler`.

## I2C/SPI Devices

I2C buses exposed as `/dev/i2c-0`, `/dev/i2c-1`, `/dev/i2c-2`, `/dev/i2c-4`:

- `i2c-0`, address `0x1e`: `gc2145` RGB sensor.
- `i2c-1`, address `0x7f`: `opn87xx_mipi` ToF/depth sensor.
- `i2c-2`, address `0x5d`: `gt911` Goodix touch controller entry from the base board/device tree.
- `i2c-4`, address `0x51`: `pcf8563` RTC.

SPI:

- `/dev/spidev1.0` and `/dev/spidev1.1` are present.
- Device tree lists both as generic `spidev` nodes on `spi1`.
- `testisp` strings reference `/dev/spidev1.1`.

## Other Hardware Interfaces

- GPIO:
  - `/dev/gpiochip0`, `/dev/gpiochip1`.
  - sysfs GPIO includes `gpio105`; `keep_app.sh` exports/toggles GPIO 105 before running the app.
- PWM:
  - `/dev/sunxi_pwm0`.
  - `/sys/class/pwm/pwmchip0` from platform device `300a000.pwm`.
- Watchdog:
  - `/dev/watchdog`, `/dev/watchdog0`.
  - Kernel says `sunxi-wdt 30090a0.watchdog`, timeout `16 sec`.
  - `keep_app.sh` feeds `/dev/watchdog` once per second.
- Graphics/video acceleration:
  - `/dev/g2d`, `/dev/ion`, `/dev/cedar_dev`, `/dev/disp`, `/dev/fb0..fb7`.
  - `testisp` links against Allwinner Eyesee-MPP/media libraries and uses `/dev/g2d`.
- Serial:
  - `/dev/ttyS0` is console from cmdline, `console=ttyS0,115200`.
  - `/dev/ttyS2` is also present.
- USB gadget:
  - Runtime interface is `usb0` at `192.168.233.1/24` using `g_ether`/CDC Ethernet.
  - Kernel log: `g_ether gadget: Ethernet Gadget`, high-speed config `CDC Ethernet (ECM)`.
  - UDC platform device: `5100000.udc-controller`.
- Network:
  - `usb0` is active for host connection.
  - `eth0` exists from platform device `5020000.eth` but is down.
  - `testisp` listens on TCP `0.0.0.0:80`; Dropbear listens on TCP `22`; `udhcpd` listens on UDP `67`.

## Userspace Firmware/App Dependencies

The main app is `/root/maix_dist/testisp`. It is an ARM hard-float musl executable and links against vendor media libraries:

- `/usr/lib/eyesee-mpp/libvenc_codec.so`
- `/usr/lib/eyesee-mpp/libvenc_base.so`
- `/usr/lib/eyesee-mpp/libvencoder.so`
- `/usr/lib/eyesee-mpp/libvdecoder.so`
- `/usr/lib/eyesee-mpp/libvideoengine.so`
- `/usr/lib/eyesee-mpp/libVE.so`
- `/usr/lib/eyesee-mpp/libMemAdapter.so`
- `/usr/lib/eyesee-mpp/libcdc_base.so`
- `/usr/lib/eyesee-mpp/libion.so`
- `/usr/lib/liblog.so`
- `/usr/lib/libglog.so.0.3.5`

Important strings in `testisp`:

- HTTP endpoints: `getdeep`, `set_cfg`, `calibration`.
- Config paths: `/mnt/cfg/CameraParms.json`, `/tmp/CameraParms.json`, `/root/maix_dist/CameraParms.json`.
- Hardware paths: `/dev/i2c-%d`, `/dev/spidev1.1`, `/dev/video0`, `/dev/video2`, `/dev/g2d`.
- It contains V4L2 capture checks and software ISP calls such as `sw_isp_driver_init`, `sw_isp_sensor_init`, `sw_isp_slave_mode`, `sw_isp_driver_start`, and `sw_isp_get_flash_bin_info`.

## Firmware Rebuild Implications

- The lowest-risk replacement path is still to keep the vendor kernel, DTB, modules, and Eyesee-MPP userspace libraries, then replace or wrap `/root/maix_dist/testisp`.
- A full firmware replacement needs equivalents for the Allwinner `sun8iw19p1`/V833 Tina kernel, device tree, boot partition, rootfs, `vin_io`/`vin_v4l2` camera stack, `opn87xx_mipi` ToF sensor driver, `gc2145` RGB sensor driver, USB Ethernet gadget setup, watchdog feeding, and likely the Eyesee-MPP/ISP libraries.
- Replacing the kernel or boot image is higher risk unless serial console/recovery is available, because the camera stack depends on vendor Allwinner video/ISP drivers and board-specific device-tree nodes.
