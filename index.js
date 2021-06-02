const express = require('express');
const fetch = require('node-fetch');
const ffmpeg = require('ffmpeg');
const fs = require('fs');
const fs_extra = require('fs-extra');
const tf = require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet');
const { createCanvas, Image } = require('canvas');
const fileUpload = require('express-fileupload');
const cors = require('cors');
const bodyParser = require('body-parser');
const morgan = require('morgan');
const _ = require('lodash');

const imageScaleFactor = 0.5;
const outputStride = 16;
const flipHorizontal = false;

const calculatePose = async (path, idx, net) => {
  let img = new Image();
  img.src = path;
  let canvas = createCanvas(img.width, img.height);
  let ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  tf.engine().startScope();
  let input = tf.browser.fromPixels(canvas);
  const pose = await net.estimateSinglePose(
    input,
    imageScaleFactor,
    flipHorizontal,
    outputStride
  );
  tf.engine().endScope();
  pose.index = idx;
  // for (let i = 0; i < pose.keypoints.length; i++) {
  //   const point = pose.keypoints[i];
  //   ctx.fillRect(Math.floor(point.position.x), Math.floor(point.position.y), 10, 10);
  // }
  // const buf = canvas.toBuffer()
  // fs.writeFileSync(path + '-part.png', buf)
  img = null;
  canvas = null;
  ctx = null;
  input = null;
  return pose;
};

const sendSuccess = (results, resultsEndpoint, token) => {
  let headers = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Token ${token}`;
  }
  return fetch(resultsEndpoint, {
    method: 'POST',
    body: JSON.stringify({
      error: false,
      results,
    }),
    headers,
  });
};
const sendError = (results, resultsEndpoint, token) => {
  let headers = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Token ${token}`;
  }
  return fetch(resultsEndpoint, {
    method: 'POST',
    body: JSON.stringify({
      error: true,
      results,
    }),
    headers,
  });
};

const processVideo = async (
  framesPath,
  filePath,
  exercise,
  points,
  token,
  resultsEndpoint
) => {
  const process = new ffmpeg(filePath);
  process.then(
    async function (video) {
      try {
        let net = await posenet.load({
          architecture: 'ResNet50',
          outputStride: 32,
          inputResolution: {
            width: video.metadata.video.resolution.w,
            height: video.metadata.video.resolution.h,
          },
          quantBytes: 2,
        });
        console.info('Posenet loaded!');
        video.fnExtractFrameToJPG(
          framesPath,
          {
            every_n_frames: 2,
            file_name: '',
          },
          async () => {
            console.info('Frames getted');
            try {
              const frames = await fs.promises.readdir(framesPath);
              const poses = [];
              for (const framename of frames) {
                console.info('Analyzing', framename);
                const pose = await calculatePose(
                  `${framesPath}/${framename}`,
                  parseInt(framename.slice(1, -4)),
                  net
                );
                poses.push(pose);
              }
              poses.sort((a, b) => a.index - b.index);
              poses_tracked_points = {
                poses: poses.map((pose) => {
                  const pose_tracked_points = [];
                  for (let i = 0; i < points.length; i++) {
                    const point = points[i];
                    const center = pose.keypoints.find(
                      (pose_point) => pose_point.part === point.center
                    );
                    const left_point = pose.keypoints.find(
                      (pose_point) => pose_point.part === point.left_point
                    );
                    const right_point = pose.keypoints.find(
                      (pose_point) => pose_point.part === point.right_point
                    );
                    const angle = find_angle(
                      left_point.position,
                      right_point.position,
                      center.position
                    );
                    if (point.min_angle) {
                      point.min_angle =
                        point.min_angle < angle ? point.min_angle : angle;
                    } else {
                      point.min_angle = angle;
                    }

                    if (point.max_angle) {
                      point.max_angle =
                        point.max_angle > angle ? point.max_angle : angle;
                    } else {
                      point.max_angle = angle;
                    }

                    pose_tracked_points.push({
                      center,
                      left_point,
                      right_point,
                      angle,
                    });
                  }
                  return {
                    index: pose.index,
                    tracked_points: pose_tracked_points,
                  };
                }),
                points,
              };
              fs.unlinkSync(filePath);
              fs.rmdirSync(framesPath, {
                recursive: true,
              });
              sendSuccess(poses_tracked_points, resultsEndpoint, token)
                .then((r) => {
                  net = null;
                  console.log(r);
                })
                .catch((r) => {
                  net = null;
                  console.log(r);
                });
            } catch (error) {
              fs.unlinkSync(filePath);
              fs.rmdirSync(framesPath, {
                recursive: true,
              });
              console.error(error);
              sendError(`Error ${error}`, resultsEndpoint, token)
                .then((r) => {
                  net = null;
                  console.log(r);
                })
                .catch((r) => {
                  net = null;
                  console.log(r);
                });
            }
          }
        );
      } catch (error) {
        console.error(error);
        fs.rmdirSync(framesPath, {
          recursive: true,
        });
      }
    },
    function (err) {
      sendError('Error: ' + err, resultsEndpoint, token)
        .then((r) => {
          console.info('response', r);
          fs.rmdirSync(framesPath, {
            recursive: true,
          });
        })
        .catch((r) => {
          console.log(r);
          fs.rmdirSync(framesPath, {
            recursive: true,
          });
        });
    }
  );
};

/*
 * Calculates the angle ABC (in radians)
 *
 * A first point, ex: {x: 0, y: 0}
 * C second point
 * B center point
 */
function find_angle(A, B, C) {
  var AB = Math.sqrt(Math.pow(B.x - A.x, 2) + Math.pow(B.y - A.y, 2));
  var BC = Math.sqrt(Math.pow(B.x - C.x, 2) + Math.pow(B.y - C.y, 2));
  var AC = Math.sqrt(Math.pow(C.x - A.x, 2) + Math.pow(C.y - A.y, 2));
  return (
    (Math.acos((BC * BC + AB * AB - AC * AC) / (2 * BC * AB)) * 180) / Math.PI
  );
}

const app = express();

app.use(
  fileUpload({
    createParentPath: true,
  })
);

//add other middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(morgan('dev'));

app.get('/', async (req, res) => {
  try {
    const process = new ffmpeg('./assets/video/test.mp4');
    process.then(
      function (video) {
        video.fnExtractFrameToJPG(
          './assets/frames/',
          {
            every_n_frames: 2,
            file_name: '',
          },
          async () => {
            console.log('frames getted :D');
            try {
              const frames = await fs.promises.readdir('./assets/frames/');
              const poses = [];
              for (const framename of frames) {
                const pose = await calculatePose(
                  `./assets/frames/${framename}`,
                  parseInt(framename.slice(1, -4))
                );
                poses.push(pose);
              }
              poses.sort((a, b) =>
                a.index > b.index ? 1 : b.index > a.index ? -1 : 0
              );
              res.json(poses);
            } catch (error) {
              console.error(error);
            }
          }
        );
      },
      function (err) {
        console.log('Error: ' + err);
      }
    );
  } catch (e) {
    console.log('error :c');
    console.log(e.code);
    console.log(e.msg);
  }
});

app.post('/video', async (req, res) => {
  try {
    if (!req.files) {
      res.send({
        status: false,
        message: 'No video uploaded',
      });
    } else {
      const file = req.files.video;
      const filePath = './assets/video/' + file.name;
      const framesPath = `./assets/frames/${file.name.replace(/\./g, '-')}`;
      fs.mkdirSync(framesPath);
      file.mv(filePath);
      console.log('File moved');
      setImmediate(() =>
        processVideo(
          framesPath,
          filePath,
          JSON.parse(req.body.exercise),
          JSON.parse(req.body.points),
          req.body.token || undefined,
          req.body.resultsEndpoint
        )
          .then(() => console.log('Proceso terminado'))
          .catch((error) => console.error(error))
      );
      console.log(req.body.token);
      res.status(200).end();
    }
  } catch (err) {
    res.status(500).send(err);
    console.error(err);
  }
});

app.listen(3000, async () => {
  console.log('Escuchando en el 3000');
});
