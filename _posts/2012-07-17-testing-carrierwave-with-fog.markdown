---
layout: post
title: "Testing Carrierwave with Fog for Amazon S3"
date: 2012-07-17 21:10
comments: true
categories: [mocking, rails, rspec, carrierwave, fog, s3]
---

Testing file upload using CarrierWave with Fog with S3 turns out to be difficult.

* CarrierWave/Fog need a non-empty file, otherwise the url to the S3 object will be nil and can't be tested.
* Fog.mock! doesn't work out of box.  Some extra steps are needed to avoid uploading files to S3 when running tests.

First, create config/fog_credentials.yml

{% highlight ruby %}
default:
  aws_access_key_id: 'your-aws-access-key-id'
  aws_secret_access_key: 'your-aws-secret-access-key/'
  region: 'your-aws-region'
{% endhighlight %}

Next, put this to your config/initializers/carrier_wave.rb

{% highlight ruby %}
Fog.credentials_path = Rails.root.join('config/fog_credentials.yml')

fog_dir = Rails.env == 'production' ? 'production-bucket' : 'dev-bucket'

CarrierWave.configure do |config|
  config.fog_credentials = {:provider => 'AWS'}
  config.fog_directory  = fog_dir
end
{% endhighlight %}

Next, put this into spec/support/fog_helper.rb

{% highlight ruby %}
Fog.mock!
Fog.credentials_path = Rails.root.join('config/fog_credentials.yml')
connection = Fog::Storage.new(:provider => 'AWS')
connection.directories.create(:key => 'dev-bucket')
{% endhighlight %}

Note, the :key value *must* match whats defined for fog_dir (dev-bucket) in carrier_wave.rb

Suppose you want to test this class:

{% highlight ruby %}
class FileUploader < CarrierWave::Uploader::Base
  storage :fog
end
{% endhighlight %}

Your rspec test can be something like this:

{% highlight ruby %}
class TestFileUploader
  mount_uploader :file, FileUploader
end

describe FileUploader do
  include FakeFS::SpecHelpers

  context 'for non-production environment' do
    it 'should upload video clip to dev-bucket on s3' do
      FakeFS.activate!
      FakeFS::File.should_receive(:chmod) #this is needed or you will get an exception
      File.open('test_file', 'w') do |f|
        f.puts('foo') # this is required or uploader_test.file.url will be nil
      end
      uploader_test = TestFileUploader.new
      uploader_test.file = File.open('test_file')
      uploader_test.save!
      uploader_test.file.url.should match /.*\/dev-bucket.*/ #test to make sure that it is not production-bucket
      FakeFS.deactivate!
    end
  end
end
{% endhighlight %}

Now the test is complete.  It uses fakefs to generate a fake file which is non-empty.  Fog will pretent to upload the file
using the FileUploader under test.  The upload url is the subject under test.