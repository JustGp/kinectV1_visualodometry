from setuptools import find_packages, setup

package_name = 'KinectV1Publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gael',
    maintainer_email='gael@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        "video_publisher = KinectV1Publisher.video_publisher:main",
        "depth_publisher = KinectV1Publisher.depth_publisher:main",
        "video_and_depth_publisher = KinectV1Publisher.video_and_depth_publisher:main",
        "kamikaze =  KinectV1Publisher.kamikaze:main",
        "video_vo = KinectV1Publisher.video_vo:main",
        "depth_vo = KinectV1Publisher.depth_vo:main"
        ],
    },
)
