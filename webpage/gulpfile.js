const $ = require('gulp');
const $changed = require('gulp-changed');
const $htmlmin = require('gulp-htmlmin');
const $merge = require('gulp-merge');
const $plumber = require('gulp-plumber');
const $postcss = require('gulp-postcss');
const $replace = require('gulp-replace');
const $filter = require('gulp-filter');

const del = require('del');
const server = require('browser-sync').create();

const paths = {
  html: {
    src: ['src/**/*.html', '!src/**/nn.html'],
    dest: 'build'
  },
  css: {
    src: 'src/**/*.css',
    dest: 'build'
  },
  img: {
    src: 'src/img/*',
    dest: 'build/img'
  }
};

$.task('build', $.series(clean, $.parallel(html, css, img)));
$.task('default', $.series('build', serve, watch));
$.task('publish', $.series('build', publish));

function clean() {
  return del(['build']);
}

function serve(done) {
  server.init({files: ['build'], server: 'build'});
  done();
}

function reload(done) {
  server.reload();
  done();
}

function watch() {
  $.watch(paths.html.src, $.series(html, reload));
  $.watch(paths.css.src, $.series(css, reload));
  $.watch(paths.img.src, $.series(img, reload));
}

function html() {
  const f = $filter('src/nn_bib.html', {restore: true});

  return $.src(paths.html.src)
    .pipe($changed(paths.html.dest))
    .pipe($plumber())
    .pipe($htmlmin({
      removeComments: true,
      collapseWhitespace: true,
      removeEmptyElements: true,
      removeStyleLinkTypeAttributes: true,
      removeEmptyAttributes: true}))
    .pipe(f)
    .pipe($replace('nn.html', 'index.html'))
    .pipe(f.restore)
    .pipe($.dest(paths.html.dest));
}

function css() {
  return $.src(paths.css.src)
    .pipe($changed(paths.css.dest))
    .pipe($plumber())
    .pipe($postcss([require('precss'),
                    require('autoprefixer'),
                    require('cssnano')]))
    .pipe($.dest(paths.css.dest));
}

function img() {
  return $.src(paths.img.src)
    .pipe($changed(paths.img.dest))
    .pipe($.dest(paths.img.dest));
}

function publish() {
  return $.src('./build/**/*').pipe($.dest('../docs'));
}
